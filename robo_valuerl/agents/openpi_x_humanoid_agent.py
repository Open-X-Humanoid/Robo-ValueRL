import torch
import numpy as np
import openpi.models.model as _model
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.models_pytorch.x_humanoid_pi0_pytorch_consistency
import openpi.models_pytorch.x_humanoid_pi0_pytorch_consistency_online_with_zero_residual
import safetensors.torch
import torch.nn as nn
import os
import jax
class OpenPi_X_Humanoid_Agent(nn.Module):
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
    

        if model_cfg.use_zero_residual_online:
            model = openpi.models_pytorch.x_humanoid_pi0_pytorch_consistency_online_with_zero_residual.X_Humanoid_PI0PytorchConsistencyOnlineZeroResidual(model_cfg).to(device)  
        elif model_cfg.consistency:
            model = openpi.models_pytorch.x_humanoid_pi0_pytorch_consistency.X_Humanoid_PI0PytorchConsistency(model_cfg).to(device)
        else:
            model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    
        if config.pytorch_weight_path is not None:
            print(f"Loading weights from: {config.pytorch_weight_path}")

            model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
            
            model_parameters = safetensors.torch.load_file(model_path)

            if config.only_load_paligemma:
                load_parameters = {key: model_parameters[key] for key in model_parameters.keys() if "paligemma" in key}
            else:
                load_parameters = model_parameters
            model.load_state_dict(load_parameters, strict=False)
            # safetensors.torch.load_model(
            #     (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model),
            #     model_path,
            #     strict=False
            # )
            print(f"Loaded PyTorch weights from {config.pytorch_weight_path}")
            # If the checkpoint is missing target-related weights (older checkpoint version),
            # sync from the corresponding student to avoid target staying at random initialization
            missing_keys = set(model.state_dict().keys()) - set(load_parameters.keys())
            if hasattr(model, 'target_action_out_proj'):
                if any('target_action_out_proj' in k for k in missing_keys):
                    with torch.no_grad():
                        for s_p, t_p in zip(model.action_out_proj.parameters(),
                                            model.target_action_out_proj.parameters()):
                            t_p.data.copy_(s_p.data)
                    print("Synced target_action_out_proj from action_out_proj (not found in checkpoint)")
            if hasattr(model, 'paligemma_with_expert') and hasattr(model.paligemma_with_expert, 'target_model'):
                if any('paligemma_with_expert.target_model' in k for k in missing_keys):
                    with torch.no_grad():
                        for s_p, t_p in zip(model.paligemma_with_expert.gemma_expert.parameters(),
                                            model.paligemma_with_expert.target_model.parameters()):
                            t_p.data.copy_(s_p.data)
                    print("Synced paligemma_with_expert.target_model from gemma_expert (not found in checkpoint)")
        
        self.model = model
        # self.prepare_preprocess()
        self.data_transforms = data_transforms
        self.image_transforms = image_transforms
        self.obs_dim = self.model.action_expert_config.width
        self.action_dim = 16
        self.chunk_size = chunk_size
    
    def get_trainable_parameters(self):
        trainable_parameters = {
            "model": self.model.parameters(),
        }
        return trainable_parameters
    
    # def prepare_preprocess(self):
    #     tokenize_transform = _transforms.TokenizePrompt(
    #         PaligemmaTokenizer(max_len=self.config.model.max_token_len)
    #     )
        
    #     resize_transform = _transforms.ResizeImages(224, 224)
    #     # resize_transform = _transforms.ResizeImages(336, 336)
    #     pad_states_and_actions = _transforms.PadStatesAndActions(self.config.model.action_dim)
    #     self.resize_transform = resize_transform
    #     self.pad_states_and_actions = pad_states_and_actions
    #     self.tokenize_transform = tokenize_transform
    #     image_transform = A.Compose([
    #     A.CenterCrop(height=224, width=224, p=1.0),
    #     ])
    #     self.image_transform = image_transform
        
    def preprocess_observation(self, batch):
        # import pdb; pdb.set_trace()
        if self.data_transforms is not None:
            # print("the batch is:", batch)
            batch = self.data_transforms(batch)
            batch['tokenized_prompt'] = torch.from_numpy(np.expand_dims(batch['tokenized_prompt'], axis=0)).to(self.device)
            batch['tokenized_prompt_mask'] = torch.from_numpy(np.expand_dims(batch['tokenized_prompt_mask'], axis=0)).to(self.device)

        # print("the transformed actions is:", batch['actions'][0])

        if self.image_transforms is not None:
            for key in batch.keys():
                if 'image' in key and 'mask' not in key:
                    # print("the key is:", key)
                    for sub_key in batch[key].keys():
                        # print("the sub_key is:", sub_key)
                        # print("the sub_key is:", sub_key)
                        # print("the data[key][sub_key] is:", data[key][sub_key].shape)
                        # import pdb; pdb.set_trace()   
                        if len(batch[key][sub_key].shape) == 4:
                            image = batch[key][sub_key].squeeze(0)
                        else:
                            image = batch[key][sub_key]
                        image = image.transpose(1,2,0)
                        transformed_image = self.image_transforms(image = image)['image']
                        transformed_image = transformed_image.transpose(2,0,1)
                        # print("the transformed_image is:", transformed_image.shape)
                        if len(batch[key][sub_key].shape) == 4:
                            batch[key][sub_key] = transformed_image[None, :, :, :]
                        else:
                            batch[key][sub_key] = transformed_image

        observation = _model.Observation.from_dict(batch)
        observation = jax.tree.map(lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else torch.from_numpy(x).to(self.device), observation)
        
        # for key in observation.images.keys():
        #     observation.images[key] = observation.images[key].permute(0, 3, 1, 2)
        
        return observation
    
    
    def forward(self, batch):
        target_device = self.device
        target_dtype = next(self.model.parameters()).dtype 

        # 2. Process Ground Truth Actions
        ground_truth_actions = batch['actions']
        if not isinstance(ground_truth_actions, torch.Tensor):
            ground_truth_actions = torch.from_numpy(ground_truth_actions)
        
        ground_truth_actions = ground_truth_actions.to(device=target_device, dtype=target_dtype)
        # print("the ground_truth_actions is:", ground_truth_actions.shape)
        # observation = self.preprocess_observation(batch)
        observation = _model.Observation.from_dict(batch)
        observation = jax.tree.map(lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else torch.from_numpy(x).to(self.device), observation)
        action_loss, consistency_loss = self.model.forward(observation, ground_truth_actions)
        return {
            "action_loss": action_loss,
            "consistency_loss": consistency_loss,
        }
    
    @torch.no_grad()
    def predict_action(self, obs):
        # print("the obs state is:", obs['state'])
        # print("the obs actions is:", obs['actions'])
        observation = self.preprocess_observation(obs)
        # print("the transformed observation state is:", observation.state)


        if isinstance(self.model, openpi.models_pytorch.x_humanoid_pi0_pytorch_consistency_subgoal.X_Humanoid_PI0PytorchConsistencyWithSubgoal):
            action, subgoal_prediction_logits = self.model.sample_actions(self.device, observation)
        else:
            action = self.model.sample_actions(self.device, observation)
        # print("the action is:", action.shape)
        # padded_action = F.pad(action, (0, 27), "constant", 0)
        # prediction = {
        #         "actions": padded_action.cpu().numpy(),
        #         "state": padded_action.cpu().numpy(),
        # }
        # result = self.unnormalize(prediction)
        # result = prediction
        action = action.cpu().numpy()
        predicted_action = action[0,:self.chunk_size, :16]
        # predicted_action[:,-1] = np.clip(predicted_action[:,-1], 0, 1)
        # predicted_action[:,7] = np.clip(predicted_action[:,7], 0, 1)
        if isinstance(self.model, openpi.models_pytorch.x_humanoid_pi0_pytorch_consistency_subgoal.X_Humanoid_PI0PytorchConsistencyWithSubgoal):
            return predicted_action, subgoal_prediction_logits
        else:
            return predicted_action


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

        # Save using safetensors, consistent with how it was loaded during initialization
        safetensors.torch.save_model(self.model, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, path, device=None):
        """
        Load model weights from the specified path.
        :param path: weights folder path (containing model.safetensors)
        :param device: the device to load onto (e.g. 'cuda' or 'cpu')
        """
        if device is not None:
            self.device = torch.device(device)
            self.model.to(self.device)

        model_path = os.path.join(path, "model.safetensors")
        if not os.path.exists(model_path):
            print(f"Error: Weights not found at {model_path}")
            return

        print(f"Loading weights from: {model_path}")
        # Load weights file
        model_parameters = safetensors.torch.load_file(model_path, device=str(self.device))

        # Load into the model
        # strict=False allows partial weight loading (e.g. only load backbone or ignore mismatched layers)
        self.model.load_state_dict(model_parameters, strict=False)
        print(f"Model loaded successfully on {self.device}")