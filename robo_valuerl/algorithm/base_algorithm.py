import torch
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import Module

class BaseAlgorithm(Module):
    def __init__(self, agent, algo_cfg, is_ddp: bool, local_rank: int):
        super().__init__()
        self.cfg = algo_cfg
        
        # Check whether wrapped by DDP
        self.is_ddp = is_ddp
        self.local_rank = local_rank

        obs_dim = agent.obs_dim
        action_dim = agent.action_dim
        chunk_size = agent.chunk_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        self.trainable_parameters = agent.get_trainable_parameters()
        if self.is_ddp:
            self.agent = DDP(agent, device_ids=[local_rank],
                             # If gradient checkpointing is enabled, this sometimes needs to be True, but try False first
            find_unused_parameters=False, 
            # Enable static graph optimization, works better with gradient checkpointing
            static_graph=True
    )
            
        else:
            self.agent = agent
        

    def step(self, batch):
        raise NotImplementedError("Subclasses must implement this method")

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        
        # Choose the correct save method depending on whether this is a DDP model
        if self.is_ddp:
            # For a DDP model, save the state of .module
            self.agent.module.save_model(path)
        else:
            # For a regular model, save directly
            self.agent.save_model(path)
            
        optimizer_path = os.path.join(path, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), optimizer_path)
        scheduler_path = os.path.join(path, "scheduler.pt")
        torch.save(self.scheduler.state_dict(), scheduler_path)
    
    def load_model(self, path, device=None):
        # Choose the correct load method depending on whether this is a DDP model
        if self.is_ddp:
            # For a DDP model, load into .module
            self.agent.module.load_model(path, device)
        else:
            # For a regular model, load directly
            self.agent.load_model(path, device)
            
        optimizer_path = os.path.join(path, "optimizer.pt")
        optimizer_state = torch.load(optimizer_path, map_location=device)
        self.optimizer.load_state_dict(optimizer_state)
        scheduler_path = os.path.join(path, "scheduler.pt")
        scheduler_state = torch.load(scheduler_path, map_location=device)
        self.scheduler.load_state_dict(scheduler_state)


