from algorithm.base_algorithm import BaseAlgorithm
import torch
class BCAlgorithm(BaseAlgorithm):
    def __init__(self, agent, algo_cfg, is_ddp: bool, local_rank: int):
        super().__init__(agent, algo_cfg, is_ddp, local_rank)

        self.optimizer_params = [{
            "params": self.trainable_parameters["encoder"],
            "lr": self.cfg.encoder_lr
        }, {
            "params": self.trainable_parameters["policy"],
            "lr": self.cfg.policy_lr
        }]
        self.optimizer = torch.optim.Adam(self.optimizer_params)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.scheduler_step_size, gamma=self.cfg.scheduler_gamma)

    def step(self, batch):
        self.optimizer.zero_grad()
        losses = self.agent.forward(batch)
        total_loss = 0
        for key,value in losses.items():
            if value is not None:
                total_loss += value
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.trainable_parameters, max_norm=1.0)
        for param_name, param_value in self.trainable_parameters.items():
            torch.nn.utils.clip_grad_norm_(param_value, max_norm=1.0)
            
        self.optimizer.step()
        self.scheduler.step()
        return losses

    def train(self, dataset):
        for batch in dataset:
            loss = self.step(batch)
            print(f"Loss: {loss}")