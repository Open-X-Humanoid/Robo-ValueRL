import torch
import numpy as np
import dataclasses
import openpi.models.model as _model
import openpi.models.pi0_config
import openpi.models_pytorch.x_humanoid_pi0_pytorch_critic_with_history
import openpi.models_pytorch.x_humanoid_pi0_pytorch_critic_without_history
import safetensors.torch
import torch.nn as nn
import os
import jax
class OpenPi_X_Humanoid_Critic(nn.Module):
    def __init__(self, config, chunk_size = 10, data_transforms = None, image_transforms = None):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.config = config
        self.chunk_size = chunk_size
        if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
            # Convert dataclass to Pi0Config if needed
            model_cfg = openpi.models.pi0_config.Pi0Config(
                dtype=config.pytorch_training_precision,
                action_dim=config.model.action_dim,
                action_horizon=config.model.action_horizon,
                max_token_len=config.model.max_token_len,
                paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
                action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
                pi05=getattr(config.model, "pi05", False),
            )
        else:
            model_cfg = config.model
            # Update dtype to match pytorch_training_precision
            object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    

        if model_cfg.use_history_image:
            model = openpi.models_pytorch.x_humanoid_pi0_pytorch_critic_with_history.X_Humanoid_PI0Pytorch_History_Critic(model_cfg).to(device)
        else:
            model = openpi.models_pytorch.x_humanoid_pi0_pytorch_critic_without_history.X_Humanoid_PI0Pytorch_Critic(model_cfg).to(device)

        self.model = model
        self.obs_dim = self.model.action_expert_config.width
        self.action_dim = 32
        self.chunk_size = chunk_size



    
        if config.pytorch_weight_path is not None:
            print(f"Loading weights from: {config.pytorch_weight_path}")

            model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
            
            model_parameters = safetensors.torch.load_file(model_path)

            if config.only_load_paligemma:
                load_parameters = {key: model_parameters[key] for key in model_parameters.keys() if "paligemma." in key}
                print("the load_parameters is:", load_parameters.keys())
            else:
                load_parameters = model_parameters
            model.load_state_dict(load_parameters, strict=False)
            # safetensors.torch.load_model(
            #     (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model), 
            #     model_path,
            #     strict=False
            # )
            print(f"Loaded PyTorch weights from {config.pytorch_weight_path}")
        
        # self.prepare_preprocess()
        self.data_transforms = data_transforms
        self.image_transforms = image_transforms
    
    def get_trainable_parameters(self):
        trainable_parameters = {
            "model": self.model.parameters(),
        }
        return trainable_parameters
    
        
    def preprocess_observation(self, batch):
        # import pdb; pdb.set_trace()
        if self.data_transforms is not None:
            batch = self.data_transforms(batch)
            batch['tokenized_prompt'] = torch.from_numpy(np.expand_dims(batch['tokenized_prompt'], axis=0)).to(self.device)
            batch['tokenized_prompt_mask'] = torch.from_numpy(np.expand_dims(batch['tokenized_prompt_mask'], axis=0)).to(self.device)

        if self.image_transforms is not None:
            for key in batch.keys():
                if 'image' == key:
                    # print("the key is:", key)
                    for sub_key in batch[key].keys():
                        # print("the sub_key is:", sub_key)
                        original_shape = batch[key][sub_key].shape
                        # import pdb; pdb.set_trace()
                        if len(original_shape) == 4:
                            image = batch[key][sub_key].squeeze(0)
                        else:
                            image = batch[key][sub_key]
                        # image = image.transpose(1,2,0)
                        transformed_image = self.image_transforms(image = image)['image']
                        # transformed_image = transformed_image.transpose(2,0,1)
                        if len(original_shape) == 4:
                            batch[key][sub_key] = transformed_image[None, :, :, :]
                        else:
                            batch[key][sub_key] = transformed_image

                if 'history_image' == key:
                    image = batch[key]
                    if len(image.shape) == 4:
                        image = image.squeeze(0)
                    else:
                        image = image
                    transformed_image = self.image_transforms(image = image)['image']
                    # trun to 4 dim, it is numpy
                    transformed_image = np.expand_dims(transformed_image, axis=0)
                    batch[key] = transformed_image


        observation = _model.Observation.from_dict(batch)
        observation = jax.tree.map(lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else torch.from_numpy(x).to(self.device), observation)
        
        # Permute images from HWC to CHW
        for key in observation.images.keys():
            observation.images[key] = observation.images[key].permute(0, 3, 1, 2)
        
        # Permute history_image from HWC to CHW (need dataclasses.replace since Observation is frozen)
        if observation.history_image is not None:
            observation = dataclasses.replace(
                observation, 
                history_image=observation.history_image.permute(0, 3, 1, 2)
            )
        
        return observation
    
    def forward(self, batch):
        ground_truth_actions = batch['actions']
        # value_gt = batch['value_gt']
        # print("the value_gt is:", value_gt.shape)
        # print("the ground_truth_actions is:", ground_truth_actions.shape)
        # observation = self.preprocess_observation(batch)
        observation = _model.Observation.from_dict(batch)
        observation = jax.tree.map(lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else torch.from_numpy(x).to(self.device), observation)

        # for key in observation.images.keys():
        #     observation.images[key] = observation.images[key].permute(0, 3, 1, 2)
            
        remain_time_estimation_loss = self.model.forward(observation, ground_truth_actions)
        return {
            "remain_time_estimation_loss": remain_time_estimation_loss,
        }

    @torch.no_grad()
    def predict_value(self, obs):
        observation = self.preprocess_observation(obs)
        remain_time_distribution = self.model.predict_remain_time(observation)
        return remain_time_distribution


    def get_observation_features(self, obs):
        observation = self.preprocess_observation(obs)
        observation_features = self.model.get_observation_features(observation)
        return observation_features

    def save_model(self, path):
        """
        Save model weights to the specified path.
        """
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        
        save_path = os.path.join(path, "model.safetensors")
        # Get the model's state_dict
        # state_dict = self.model.state_dict()
        
        # Save using safetensors, consistent with the loading method used at initialization
        safetensors.torch.save_model(self.model, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, path, device=None):
        """
        Load model weights from the specified path.
        :param path: Path to the weights folder (containing model.safetensors)
        :param device: Device to load onto (e.g. 'cuda' or 'cpu')
        """
        if device is not None:
            self.device = torch.device(device)
            self.model.to(self.device)

        model_path = os.path.join(path, "model.safetensors")
        if not os.path.exists(model_path):
            print(f"Error: Weights not found at {model_path}")
            return

        print(f"Loading weights from: {model_path}")
        # Load the weights file
        model_parameters = safetensors.torch.load_file(model_path, device=str(self.device))
        
        # Load into the model
        # strict=False allows loading partial weights (e.g. only the backbone, or ignoring mismatched layers)
        self.model.load_state_dict(model_parameters, strict=False)
        print(f"Model loaded successfully on {self.device}")