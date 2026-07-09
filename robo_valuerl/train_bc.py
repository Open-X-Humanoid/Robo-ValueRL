import os
import time
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import albumentations as A
from tqdm import tqdm
import wandb
import json
# Your custom modules
import openpi.training.config as _config
from agents.openpi_x_humanoid_agent import OpenPi_X_Humanoid_Agent
from transforms.tokenizer import PaligemmaTokenizer
from dataset.x_humanoid_lerobot_offline_wrapper import XHumanoidLerobotOfflineWrapperForBC
import transforms.transforms as _transforms
from config.factory import FilteredBCPIAlgorithmFactory
from config.algorithm_cfg import FilteredBCPIAlgorithmConfig
import openpi.models.model as _model
import torch
import numpy as np
from torch.utils.data import DataLoader
from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset
from openpi.models.model import ModelType

def is_ddp_running():
    """Check whether running in a DDP environment"""
    return "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ

def recursive_to_device(data, device, non_blocking=True):
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {k: recursive_to_device(v, device, non_blocking) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_to_device(v, device, non_blocking) for v in data]
    elif isinstance(data, tuple):
        return tuple(recursive_to_device(v, device, non_blocking) for v in data)
    else:
        return data

def setup_ddp():
    """Initialize the DDP process group"""
    if not is_ddp_running():
        return None, None, False
    
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, world_size, True

def cleanup_ddp():
    """Clean up the DDP process group"""
    if dist.is_initialized():
        dist.destroy_process_group()

def setup_single_gpu():
    """Set up single-GPU training"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        return device, 0, False
    else:
        device = torch.device("cpu")
        return device, 0, False

def recursive_detach(data):
    """
    Core optimization: detach all Tensors in the dict from the computation graph and convert them to scalars.
    This is key to preventing GPU 0 memory blowup.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().item()
    elif isinstance(data, dict):
        return {k: recursive_detach(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_detach(v) for v in data)
    return data

# --- Main training logic ---
def main_worker(local_rank=None, world_size=None, is_ddp=False):
    # 1. Initialize device
    if is_ddp:
        torch.cuda.set_device(local_rank) # must be done first
        device = torch.device(f"cuda:{local_rank}")
        is_main = (local_rank == 0)
        is_main_computer = (dist.get_rank() == 0)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        world_size = 1
        is_main = True

        is_main_computer = True
    config = _config.cli()
    algo_config = FilteredBCPIAlgorithmConfig(decay_steps = config.decay_steps, lr = config.lr)

    # 2. Adjust batch size: ensure the total batch size matches expectations
    # Assume config.batch_size is the total batch size you want
    per_device_batch_size = max(1, config.batch_size // world_size)

    if is_main:
        wandb.init(
            project=config.project_name if hasattr(config, 'project_name') else "vla_project",
            name=config.exp_name if hasattr(config, 'exp_name') else f"ddp_rank_{time.strftime('%m%d_%H%M')}",
            config={"actual_per_gpu_batch": per_device_batch_size, "world_size": world_size}
        )

    # 3. Model initialization (load directly onto the corresponding GPU)
    norm_stats = json.load(open(config.data.norm_stats_path))
    denorm_stats = norm_stats.copy()
    # change_gripper_action = _transforms.ChangeGripperAction()  
    # Truncate stats according to model action_dim (16-bit or 32-bit)
    target_dim = config.model.action_dim
    for key in norm_stats.keys():
        norm_stats[key] = {
            "mean": np.array(norm_stats[key]["mean"][:target_dim]),
            "std": np.array(norm_stats[key]["std"][:target_dim]),
            # "q01": np.array(norm_stats[key]["q01"][:target_dim]),
            # "q99": np.array(norm_stats[key]["q99"][:target_dim]),
        }
    
    # Stats used for denormalization (usually take the first 16 dims for humanoid joint control)
    for key in denorm_stats.keys():
        denorm_stats[key] = {
            "mean": np.array(denorm_stats[key]["mean"][:16]),
            "std": np.array(denorm_stats[key]["std"][:16]),
            # "q01": np.array(denorm_stats[key]["q01"][:16]),
            # "q99": np.array(denorm_stats[key]["q99"][:16]),
        }
    norm_state_action = _transforms.NormalizeStatesActions(
        norm_stats, use_quantile_norm=config.model.model_type != ModelType.PI0
    )
    mask = [True] * config.model.action_dim
    mask[7] = False
    mask[-1] = False
    
    transforms = _transforms.compose([
        _transforms.PadStatesAndActions(config.model.action_dim),
        _transforms.DeltaAction(mask=mask),
        norm_state_action,
        _transforms.InjectDefaultPrompt(prompt="separate the blocks and sort by color into different plates"),
        _transforms.ResizeImages(256, 256),
        _transforms.TokenizePrompt(PaligemmaTokenizer(max_len=200 if config.model.pi05 else 48)),
    ])

    image_transforms = A.Compose([
    # A.RandomResizedCrop(size=(224, 224), scale=(0.75, 1.0), p=1.0),
    A.RandomCrop(height=224, width=224, p=1.0),
    # A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    ], p=0.8),
    A.CLAHE(clip_limit=4.0, p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3), 
    A.GaussNoise(std_range=(0.02, 0.05), p=0.2),
    A.CoarseDropout(
        num_holes_range=(1, 4), 
        hole_height_range=(10, 20), 
        hole_width_range=(10, 20), 
        p=0.3
    ),
    ])

    agent = OpenPi_X_Humanoid_Agent(config, chunk_size=config.chunk_size, data_transforms=transforms)
    if hasattr(agent.model, "gradient_checkpointing_enable"):
        # Many transformers models support calling this directly
        agent.model.gradient_checkpointing_enable()
        print("Gradient Checkpointing enabled for the model.")
    elif hasattr(agent, "model") and hasattr(agent.model, "set_grad_checkpointing"):
        # Some timm models or custom models use this
        agent.model.set_grad_checkpointing(enable=True)
    
    agent = agent.to(device)
    
    # Ensure the model has been dispatched to the corresponding GPU before creating the algorithm object
    algorithm = FilteredBCPIAlgorithmFactory().create_algorithm(agent, algo_config, is_ddp, local_rank)

    # 4. Data loader optimization
    offline_delta_timestamps = {key : [t / 30 for t in range(config.chunk_size)] for key in config.action_sequence_keys}
    
    # if config.model.use_action_quality_filter:
    offline_delta_timestamps.update({"quality": [t / 30 for t in range(config.chunk_size)]})
    
    offline_dataset = MultiLeRobotDataset(repo_ids=config.data.repo_ids, root=config.data.root, delta_timestamps=offline_delta_timestamps)   
    offline_wrapper = XHumanoidLerobotOfflineWrapperForBC(offline_dataset, sample_from_ratio=config.sample_from_ratio, transforms=transforms, image_transform=image_transforms)
    
    # online_buffer = RealWorldOnlineBuffer(temp_dir="./tmp/no", data_spec=X_HUMANOID_Online_DATA_SPEC, 
                                    # transforms=transforms, buffer_capacity = 10000, fps =30)
    # combined_dataset = ConcatDataset([offline_wrapper, online_buffer])
    combined_dataset = offline_wrapper

    sampler = DistributedSampler(combined_dataset, num_replicas=world_size, rank=local_rank, shuffle=True) if is_ddp else None
    
    dataloader = DataLoader(
        combined_dataset, 
        sampler=sampler, 
        batch_size=per_device_batch_size, 
        num_workers=8,        # too high a num_workers increases CPU memory pressure, indirectly affecting GPU communication
        pin_memory=True,     # must be used together with non_blocking=True
        drop_last=True
    )

    # 5. Training loop optimization
    global_step = 0
    final_checkpoint_dir = config.checkpoint_dir / time.strftime("%Y%m%d_%H%M%S")
    if is_main_computer: final_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.num_epochs):
        if is_ddp: sampler.set_epoch(epoch)
        
        progress_bar = tqdm(
            enumerate(dataloader), 
            total=len(dataloader), 
            disable=not is_main,
            desc=f"Epoch {epoch}"
        )
        
        for idx, batch in progress_bar:
            # A. Asynchronously move data
            batch = recursive_to_device(batch, device, non_blocking=True)
            
            # print("the batch is:", batch["actions"][:,7])
            # B. Training step
            loss_dict_raw = algorithm.step(batch)
            
            # C. Memory management: immediately detach from the computation graph
            # Keep only the values for logging, drop the references to the Tensors
            metrics = recursive_detach(loss_dict_raw)
            
            global_step += 1

            # D. Logging (main process only)
            if is_main:
                total_loss = sum([v for v in metrics.values() if isinstance(v, (int, float))])
                
                if global_step % 10 == 0:
                    progress_bar.set_postfix({
                        "L": f"{total_loss:.3f}", 
                        "step": global_step
                    })

                if global_step % 50 == 0:
                    wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=global_step)

            # E. Explicit cleanup
            # This step is very useful when memory is extremely tight
            del batch, loss_dict_raw
            
            # Periodically clean up memory fragmentation (not recommended every step, it slows things down)
            if global_step % 500 == 0:
                torch.cuda.empty_cache()

            # F. Save and stop logic
            if is_main_computer and global_step % algo_config.save_interval_steps == 0:
                algorithm.save_model(final_checkpoint_dir / f"step_{global_step}")

            if global_step >= algo_config.decay_steps:
                break
        
        if global_step >= algo_config.decay_steps: break

    if is_ddp: dist.barrier()
        
def main():
    """Unified entry point"""
    local_rank, world_size, is_ddp = setup_ddp()
    
    try:
        if is_ddp:
            main_worker(local_rank, world_size, True)
        else:
            main_worker()
    finally:
        if not is_ddp or (is_ddp and local_rank == 0):
            wandb.finish()
        cleanup_ddp()

if __name__ == "__main__":
    main()