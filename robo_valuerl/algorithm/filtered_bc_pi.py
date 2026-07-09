import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from algorithm.base_algorithm import BaseAlgorithm

class FilteredBCPIAlgorithm(BaseAlgorithm):
    def __init__(self, agent, algo_cfg, is_ddp: bool, local_rank: int):
        super().__init__(agent, algo_cfg, is_ddp, local_rank)

        # 1. Extract config parameters (assumes these parameters are already defined in self.cfg)
        # If the self.cfg structure differs, adjust the reference paths accordingly
        peak_lr = self.cfg.lr  # or use config.lr_schedule.peak_lr
        warmup_steps = self.cfg.warmup_steps
        decay_steps = self.cfg.decay_steps
        end_lr = self.cfg.decay_lr
        
        # 2. Create the AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            self.trainable_parameters["model"],
            lr=peak_lr,
            betas=(self.cfg.optimizer_b1, self.cfg.optimizer_b2),
            eps=self.cfg.optimizer_eps,
            weight_decay=self.cfg.optimizer_weight_decay,
            fused=True
            
        )

        # 3. Define the learning rate schedule logic (lambda function)
        # LambdaLR's return value is a multiplier of base_lr. For convenience, we compute the absolute LR and then divide by peak_lr
        def lr_lambda(step: int):
            if step < warmup_steps:
                # Linear warmup phase
                init_lr = peak_lr / (warmup_steps + 1)
                current_lr = init_lr + (peak_lr - init_lr) * step / warmup_steps
            else:
                # Cosine annealing phase
                progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
                cos_val = 0.5 * (1 + np.cos(np.pi * progress))
                current_lr = end_lr + (peak_lr - end_lr) * cos_val
            
            # Return the current learning rate as a coefficient relative to the initial learning rate (peak_lr)
            return current_lr / peak_lr

        # 4. Initialize the scheduler
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        
        # losses = self.agent.forward(batch)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            losses = self.agent.forward(batch)
            
            total_loss = 0
            for key, value in losses.items():
                if value is not None:
                    total_loss += value
                    

        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.trainable_parameters["model"], max_norm=3.0)

        self.optimizer.step()

        # Update the target network of the consistency model (EMA)
        # Note: under DDP, the original model must be accessed via .module
        model_to_update = self.agent.module if self.is_ddp else self.agent
        if hasattr(model_to_update.model, 'update_target_model'):
            model_to_update.model.update_target_model()

        # scheduler.step() here increments the internal step count
        self.scheduler.step()
        
        # Optional: record the current learning rate for debugging
        current_lr = self.optimizer.param_groups[0]['lr']
        losses['lr'] = current_lr
        
        return losses

