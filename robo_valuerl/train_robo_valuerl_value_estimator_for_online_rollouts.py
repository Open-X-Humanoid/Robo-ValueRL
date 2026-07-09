import os
import time
import dataclasses
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
import albumentations as A
from tqdm import tqdm
import wandb
from agents.openpi_x_humanoid_critic import OpenPi_X_Humanoid_Critic
# Assume these are your custom modules
import openpi.training.config as _config
from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset
# from dataset.x_humanoid_lerobot_offline_wrapper import XHumanoidLerobotOfflineWrapperForBC
import transforms.transforms as _transforms
from transforms.tokenizer import PaligemmaTokenizer
from config.factory import FilteredBCPIAlgorithmFactory
from config.algorithm_cfg import FilteredBCPIAlgorithmConfig
from dataset.x_humanoid_other_value_function_lerobot_offline_wrapper import OtherXHumanoidValueFunctionWithPreHistoryLerobotOfflineWrapperForBC, OtherOpenXValueFunctionWithPreHistoryLerobotOfflineWrapperForBC
import torch
from dataset.x_humanoid_value_function_lerobot_offline_wrapper import XHumanoidValueFunctionWithPreHistoryLerobotOfflineWrapperForBC
from dataset.proportional_multi_dataset_sampler import BatchMixedProportionalSampler
# --- Optimized utility functions ---

def recursive_to_device(data, device, non_blocking=True):
    """
    When using non_blocking=True, pin_memory=True must also be set to truly achieve an asynchronous copy
    """
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {k: recursive_to_device(v, device, non_blocking) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_to_device(v, device, non_blocking) for v in data)
    return data

def recursive_detach_and_cpu(data):
    """
    Core optimization logic:
    1. .detach() cuts off the reference to the computation graph
    2. .cpu() moves data from GPU memory to host memory
    3. .item() converts to a Python scalar, fully relieving GPU 0 of the extra burden of logging
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().item()
    elif isinstance(data, dict):
        return {k: recursive_detach_and_cpu(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_detach_and_cpu(v) for v in data)
    return data

def setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        return None, None, False
    
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank) # Ensure each process only sees its own GPU
    return local_rank, world_size, True

# --- Main training logic ---

def main_worker(local_rank=None, world_size=None, is_ddp=False):
    # 1. Set up CUDA context first to prevent process confusion
    if is_ddp:
        device = torch.device(f"cuda:{local_rank}")
        is_main = (local_rank == 0)
        is_main_computer = (dist.get_rank() == 0)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        local_rank, world_size, is_main = 0, 1, True
        is_main_computer = True

    config = _config.cli()
    algo_config = FilteredBCPIAlgorithmConfig(decay_steps=config.decay_steps, lr=config.lr)

    # 2. Launch WandB only on the main process
    if is_main:
        wandb.init(
            project=config.project_name if hasattr(config, 'project_name') else "vla_project",
            name=config.exp_name if hasattr(config, 'exp_name') else f"ddp_rank_{time.strftime('%m%d_%H%M')}",
            config=dataclasses.asdict(config) if dataclasses.is_dataclass(config) else {}
        )

    # 3. Initialize transforms
    transforms = _transforms.compose([
        _transforms.InjectDefaultPrompt(prompt="separate the blocks and sort by color into different plates"),
        _transforms.TokenizePrompt(PaligemmaTokenizer(max_len=200)),
        _transforms.ResizeImages(256, 256),
        _transforms.PadStatesAndActions(config.model.action_dim),
        # _transforms.ConvertXHumanoidName()
    ])

    image_transforms = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.75, 1.0), p=1.0),
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

    # 4. Load model (memory optimization: move to device first, then wrap DDP)
    agent = OpenPi_X_Humanoid_Critic(config, chunk_size=config.chunk_size, data_transforms=transforms, image_transforms=None)
    
    # if hasattr(agent.model, "gradient_checkpointing_enable"):
    #     # Many transformers models support this direct call
    #     agent.model.gradient_checkpointing_enable()
    #     print("Gradient Checkpointing enabled for the model.")
    # elif hasattr(agent, "model") and hasattr(agent.model, "set_grad_checkpointing"):
    #     # Some timm models or custom models use this
    #     agent.model.set_grad_checkpointing(enable=True)
    
    agent = agent.to(device)
    
    # Algorithm wrapper (includes DDP wrapper)
    algorithm = FilteredBCPIAlgorithmFactory().create_algorithm(agent, algo_config, is_ddp, local_rank)

    # 5. Data pipeline optimization
    offline_delta_timestamps = {key : [t / 30 for t in range(config.chunk_size)] for key in config.action_sequence_keys}
    
    offline_delta_timestamps.update({
        "observation.image.image" : [-5 / 30, 0],
        "return_to_go": [-5 / 30,0 ]
    })
    
    print("the offline_delta_timestamps is:", offline_delta_timestamps)
    offline_dataset = MultiLeRobotDataset(repo_ids=config.data.repo_ids, root=config.data.root, delta_timestamps=offline_delta_timestamps)   
    offline_wrapper = XHumanoidValueFunctionWithPreHistoryLerobotOfflineWrapperForBC(offline_dataset, sample_from_ratio=1, transforms=transforms, image_transform=image_transforms)
    
    rollout_dataset_timestamps = {key : [t / 30 for t in range(config.chunk_size)] for key in config.action_sequence_keys}
    rollout_dataset_timestamps.update({
        "observation.image.image" : [-5 / 30, 0],
        "return_to_go": [-5 / 30,0 ]
    })
    rollout_dataset = MultiLeRobotDataset(repo_ids=config.data.rollout_repo_ids, root=config.data.rollout_root, delta_timestamps=rollout_dataset_timestamps)   
    rollout_wrapper = XHumanoidValueFunctionWithPreHistoryLerobotOfflineWrapperForBC(rollout_dataset, sample_from_ratio=1, transforms=transforms, image_transform=image_transforms)

    other_dataset_timestamps = {key : [t / 30 for t in range(config.chunk_size)] for key in ("action.arm_joint_position", "action.hand_joint_position")}
    other_dataset_timestamps.update({
        "observation.rgb_images.camera_head" : [-5 / 30, 0],
        "return_to_go": [-5 / 30,0 ]
    })
    
    other_dataset = MultiLeRobotDataset(repo_ids=config.data.filtered_bc_repo_ids, root=config.data.filtered_bc_root, delta_timestamps=other_dataset_timestamps)   
    other_wrapper = OtherXHumanoidValueFunctionWithPreHistoryLerobotOfflineWrapperForBC(other_dataset, sample_from_ratio=1, transforms=transforms, image_transform=image_transforms)
    
    
    openx_dataset_timestamps = {"action" : [t for t in range(config.chunk_size)]}
    openx_dataset_timestamps.update({
        "observation.images.image" : [-5 / 5, 0],
        "return_to_go": [-5 / 5,0 ]
    })
    openx_dataset = MultiLeRobotDataset(repo_ids=config.data.openx_repo_ids, root=config.data.openx_root, delta_timestamps=openx_dataset_timestamps)
    openx_wrapper = OtherOpenXValueFunctionWithPreHistoryLerobotOfflineWrapperForBC(openx_dataset, sample_from_ratio=1, transforms=transforms, image_transform=image_transforms)
    
    # online_buffer = RealWorldOnlineBuffer(temp_dir="./tmp/no", data_spec=X_HUMANOID_Online_DATA_SPEC, 
    #                                 transforms=transforms, buffer_capacity=10000, fps=30)
    # combined_dataset = ConcatDataset([offline_wrapper, online_buffer])
    combined_dataset = ConcatDataset([offline_wrapper, rollout_wrapper, other_wrapper, openx_wrapper])  

    # rollout accounts for 50%, offline / other / openx each account for ~16.7%
    # Ratio offline:rollout:other:openx = 1:3:1:1
    # rollout accounts for 50%, offline / other / openx each account for ~16.7%
    # Must use the smallest dataset (rollout, idx=1) as the base to avoid other datasets triggering the cap
    sampler = BatchMixedProportionalSampler(
        combined_dataset=combined_dataset,
        weights=[1, 3, 1, 1],
        base_dataset_idx=1,          # rollout as the base (smallest dataset)
        batch_size=max(1, config.batch_size // world_size),  # consistent with the DataLoader's batch_size
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
    )

    # ── Sampling ratio verification (rank 0 only, run once before training) ──────────────────────────
    if local_rank == 0:
        dataset_names = ["offline", "rollout", "other", "openx"]
        boundaries = list(combined_dataset.cumulative_sizes)  # [n0, n0+n1, ...]
        counts = [0] * len(dataset_names)
        for idx in sampler:
            for ds_i, boundary in enumerate(boundaries):
                if idx < boundary:
                    counts[ds_i] += 1
                    break
        total = sum(counts)
        print("\n[Sampler Ratio Check]")
        for name, cnt in zip(dataset_names, counts):
            print(f"  {name:10s}: {cnt:6d}  ({cnt/total*100:.1f}%)")
        print(f"  {'total':10s}: {total:6d}")
    # ──────────────────────────────────────────────────────────────────────

    #sampler = DistributedSampler(combined_dataset, shuffle=True, drop_last=True, num_replicas=world_size, rank=local_rank) if is_ddp else None
    # Memory optimization: set num_workers and persistent_workers appropriately
    dataloader = DataLoader(
        combined_dataset, 
        sampler=sampler, 
        batch_size=max(1, config.batch_size // world_size), 
        num_workers=8,
        pin_memory=True,      # Must be paired with recursive_to_device(non_blocking=True)
        persistent_workers=True,
        drop_last=True
    )

    # 6. Core training loop
    global_step = 0
    final_checkpoint_dir = config.checkpoint_dir / time.strftime("%Y%m%d_%H%M%S")
    if is_main_computer: final_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(4):
        if is_ddp: sampler.set_epoch(epoch)
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), disable=not is_main, desc=f"Epoch {epoch}")
        
        for idx, batch in progress_bar:
            # A. Asynchronously transfer data to reduce CPU-GPU wait overhead
            batch = recursive_to_device(batch, device, non_blocking=True)

            # B. Training step (gradient computation, update)
            # Note: algorithm.step must ensure gradients are cleaned up promptly after syncing on the main process
            # loss_dict_raw = algorithm.step(batch)

            loss_dict_raw = algorithm.step(batch)
            # C. Core memory management: forcibly detach the computation graph and convert to scalar
            # This step prevents the loss tensor from holding onto a huge autograd graph in GPU 0 memory
            metrics_scalar = recursive_detach_and_cpu(loss_dict_raw)
            
            global_step += 1

            # D. Logging
            if is_main:
                # Only compute the total loss on the main process
                total_loss = sum([v for v in metrics_scalar.values() if isinstance(v, (int, float))])
                
                if global_step % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{total_loss:.4f}", "Step": global_step, **metrics_scalar})

                if global_step % 50 == 0:
                    wandb.log({f"train/{k}": v for k, v in metrics_scalar.items()}, step=global_step)

            # E. Explicitly free memory (core optimization)
            # In Python, the batch variable is not released before the loop ends; explicit del combined with empty_cache
            # can greatly alleviate the GPU memory fragmentation caused by multi-process DDP.
            del batch, loss_dict_raw
            
            if global_step % 200 == 0:
                # Periodically clear fragmentation to prevent GPU 0 memory from steadily climbing
                torch.cuda.empty_cache()

            # F. Save model
            if is_main_computer and global_step % algo_config.save_interval_steps == 0:
                algorithm.save_model(final_checkpoint_dir / f"step_{global_step}")

            if global_step >= algo_config.decay_steps:
                break

        # Save checkpoint after each epoch
        if is_main_computer:
            algorithm.save_model(final_checkpoint_dir / f"epoch_{epoch}_step_{global_step}")

        if global_step >= algo_config.decay_steps: break

    # Training finished, synchronize all processes
    if is_ddp: dist.barrier()
        
def main():
    local_rank, world_size, is_ddp = setup_ddp()
    try:
        main_worker(local_rank, world_size, is_ddp)
    finally:
        if not is_ddp or (is_ddp and local_rank == 0):
            wandb.finish()
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()