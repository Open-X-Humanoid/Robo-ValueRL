import os
import time
import dataclasses
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb
# Assume these are your custom modules
import openpi.training.config as _config
from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset
# from dataset.x_humanoid_lerobot_offline_wrapper import XHumanoidLerobotOfflineWrapperForBC
import transforms.transforms as _transforms
from transforms.tokenizer import PaligemmaTokenizer
from config.algorithm_cfg import FilteredBCPIAlgorithmConfig
import torch
from torch.utils.data import DataLoader
from dataset.x_humanoid_value_function_lerobot_offline_wrapper import XHumanoidValueFunctionWithPreHistoryLerobotOfflineWrapperForBC

# --- Optimized utility functions ---

def recursive_to_device(data, device, non_blocking=True):
    """
    When using non_blocking=True, pin_memory=True must also be set to actually achieve an asynchronous copy
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
    1. .detach() cuts the reference to the computation graph
    2. .cpu() moves data from GPU memory to host memory
    3. .item() converts to a Python scalar, fully releasing the extra burden on GPU 0 during logging
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
    torch.cuda.set_device(local_rank) # ensure each process only sees its own GPU
    return local_rank, world_size, True

# --- Main training logic ---

def main_worker(local_rank=None, world_size=None, is_ddp=False):
    # 1. Set the CUDA context first to avoid process confusion
    if is_ddp:
        device = torch.device(f"cuda:{local_rank}")
        is_main = (local_rank == 0)
        is_main_computer = (dist.get_rank() == 0)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        local_rank, world_size, is_main = 0, 1, True

    config = _config.cli()
    algo_config = FilteredBCPIAlgorithmConfig()

    # 2. WandB is only initialized on the main GPU
    if is_main:
        wandb.init(
            project=config.wandb_project if hasattr(config, 'wandb_project') else "vla_project",
            name=config.wandb_run_name if hasattr(config, 'wandb_run_name') else f"ddp_{time.strftime('%m%d_%H%M')}",
            config=dataclasses.asdict(config) if dataclasses.is_dataclass(config) else {}
        )

    # 3. Initialize Transforms
    transforms = _transforms.compose([
        _transforms.InjectDefaultPrompt(prompt="separate the blocks and sort by color into different plates"),
        _transforms.TokenizePrompt(PaligemmaTokenizer(max_len=48)),
        _transforms.ResizeImages(224, 224),
        _transforms.PadStatesAndActions(config.model.action_dim),
        # _transforms.ConvertXHumanoidName()
    ])

    # 4. Model loading (memory optimization: move to device first, then wrap DDP)
    # agent = OpenPi_X_Humanoid_Critic(config, chunk_size=config.chunk_size, data_transforms=transforms, image_transforms=None)
    
    # if hasattr(agent.model, "gradient_checkpointing_enable"):
    #     # Many transformers models support calling this directly
    #     agent.model.gradient_checkpointing_enable()
    #     print("Gradient Checkpointing enabled for the model.")
    # elif hasattr(agent, "model") and hasattr(agent.model, "set_grad_checkpointing"):
    #     # Some timm models or custom models use this
    #     agent.model.set_grad_checkpointing(enable=True)
    
    # agent = agent.to(device)
    
    # Algorithm wrapper (includes DDP Wrapper internally)
    # algorithm = FilteredBCPIAlgorithmFactory().create_algorithm(agent, algo_config, is_ddp, local_rank)

    # 5. Data pipeline optimization
    offline_delta_timestamps = {key : [t / 30 for t in range(config.chunk_size)] for key in config.action_sequence_keys}
    
    offline_delta_timestamps.update({"observation.image.image" : [-1, 0]})
    
    print("the offline_delta_timestamps is:", offline_delta_timestamps)
    offline_dataset = MultiLeRobotDataset(repo_ids=config.data.repo_ids, root=config.data.root, delta_timestamps=offline_delta_timestamps)   
    offline_wrapper = XHumanoidValueFunctionWithPreHistoryLerobotOfflineWrapperForBC(offline_dataset, transforms=transforms, image_transform=None)
    
    # online_buffer = RealWorldOnlineBuffer(temp_dir="./tmp/no", data_spec=X_HUMANOID_Online_DATA_SPEC, 
    #                                 transforms=transforms, buffer_capacity=10000, fps=30)
    # combined_dataset = ConcatDataset([offline_wrapper, online_buffer])
    combined_dataset = offline_wrapper

    sampler = DistributedSampler(combined_dataset, num_replicas=world_size, rank=local_rank, shuffle=True) if is_ddp else None
    
    # Memory optimization: set num_workers and persistent_workers appropriately
    dataloader = DataLoader(
        combined_dataset, 
        sampler=sampler, 
        batch_size=max(1, config.batch_size // world_size), 
        num_workers=8,
        pin_memory=True,      # must be used together with recursive_to_device(non_blocking=True)
        persistent_workers=True,
        drop_last=True
    )

    # 6. Core training loop
    global_step = 0
    final_checkpoint_dir = config.checkpoint_dir / time.strftime("%Y%m%d_%H%M%S")
    if is_main_computer: final_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.num_epochs):
        if is_ddp: sampler.set_epoch(epoch)
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), disable=not is_main, desc=f"Epoch {epoch}")
        
        for idx, batch in progress_bar:
            # A. Asynchronously move data, reducing CPU-GPU wait/memory overhead
            batch = recursive_to_device(batch, device, non_blocking=True)
            # print("the history is:", batch['history_image'])
            # B. Training step (gradient computation, update)
            # Note: algorithm.step must ensure gradients are cleaned up promptly after syncing on the main GPU
            # loss_dict_raw = algorithm.step(batch)

            # loss_dict_raw = algorithm.step(batch)
            # C. Core of memory management: forcibly detach from the computation graph and convert to scalars
            # This step prevents the Loss Tensor from keeping a huge Autograd graph resident in GPU 0 memory
            # metrics_scalar = recursive_detach_and_cpu(loss_dict_raw)
            
            global_step += 1

            # D. Logging
            if is_main:
                # Only compute the total loss on the main GPU
                total_loss = sum([v for v in metrics_scalar.values() if isinstance(v, (int, float))])
                
                if global_step % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{total_loss:.4f}", "Step": global_step, **metrics_scalar})

                if global_step % 50 == 0:
                    wandb.log({f"train/{k}": v for k, v in metrics_scalar.items()}, step=global_step)

            # E. Explicitly free memory (core optimization)
            # In Python, the batch variable isn't released before the loop ends; explicit del combined with empty_cache
            # greatly alleviates memory fragmentation issues caused by multi-process DDP.
            del batch, loss_dict_raw
            
            if global_step % 200 == 0:
                # Periodically clean up fragmentation to avoid GPU 0 memory continuously climbing
                torch.cuda.empty_cache()

            # F. Save the model
            if is_main_computer and global_step % algo_config.save_interval_steps == 0:
                algorithm.save_model(final_checkpoint_dir / f"step_{global_step}")

            if global_step >= algo_config.decay_steps:
                break
        
        if global_step >= algo_config.decay_steps: break

    # Training finished, sync all GPUs
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