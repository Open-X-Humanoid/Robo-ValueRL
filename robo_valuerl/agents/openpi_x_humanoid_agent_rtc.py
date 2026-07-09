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
class OpenPi_X_Humanoid_Agent_RTC(nn.Module):
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
    

        print("the model_cfg is:", model_cfg)
        if model_cfg.use_zero_residual_online:
            model = openpi.models_pytorch.x_humanoid_pi0_pytorch_consistency_online_with_zero_residual.X_Humanoid_PI0PytorchConsistencyOnlineZeroResidual(model_cfg).to(device)
            self.num_steps = 1
        elif model_cfg.consistency:
            model = openpi.models_pytorch.x_humanoid_pi0_pytorch_consistency.X_Humanoid_PI0PytorchConsistency(model_cfg).to(device)
            self.num_steps = 1
        else:
            model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
            self.num_steps = 5

        print("wtf the model is:", model)

        # For LoRA: must apply LoRA BEFORE loading weights, because the checkpoint
        # was saved after PEFT wrapping (keys have base_model.model prefix).
        if model_cfg.use_lora_online:
            model.freeze_base_model()
            print("Applied LoRA adapters to model (before loading weights)")

        if config.pytorch_weight_path is not None:
            print(f"Loading weights from: {config.pytorch_weight_path}")

            model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
            model_state = safetensors.torch.load_file(model_path)
            missing, unexpected = model.load_state_dict(model_state, strict=False)
            print(f"Loaded PyTorch weights from {config.pytorch_weight_path}")
            if missing:
                print(f"  Missing keys: {len(missing)}")
                print(f"  Missing keys: {missing}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
                print(f"  Unexpected keys: {unexpected}")

        self.model = model
        # self.prepare_preprocess()
        self.data_transforms = data_transforms
        self.image_transforms = image_transforms
        self.obs_dim = self.model.action_expert_config.width
        self.action_dim = 32
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
        ground_truth_actions = batch['actions']
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

        # action, _ = self.model.get_action_with_return_to_go(self.device,observation)
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
        return predicted_action



    def predict_rtc_action_with_noise(self, obs, a_prev=None, d=0, guidance_scale=1.0):
        """
        RTC-specific inference interface

        Args:
            obs: raw observation dict
            a_prev: raw action chunk generated at the previous timestep (numpy array)
            d: predicted number of inference latency steps
            s: number of steps the robot has executed since the last inference started
        """
        # 1. Preprocess the current observation
        observation = self.preprocess_observation(obs)

        # 2. Process the reference action chunk a_prev
        a_prev_tensor = None
        if a_prev is not None:
            # RTC guidance requires actions to be in normalized space
            # Convert numpy to tensor
            if isinstance(a_prev, list):
                a_prev = np.array(a_prev)
            a_prev_torch = torch.from_numpy(a_prev).to(self.device).float()

            # [Important] If a_prev is an unnormalized physical value, it must first be normalized back via the transform
            # Here we assume a_prev has shape [H, 16] or [1, H, 16]
            if a_prev_torch.ndim == 2:
                a_prev_torch = a_prev_torch.unsqueeze(0)

            # Use the existing transform to normalize (actions only)
            # Note: this requires that your NormalizeStatesActions supports handling actions separately
            # print("the a_prev_torch is:", a_prev_torch.shape)
            # print("the a_prev_torch is:", a_prev_torch)
            tmp_data = {"actions": a_prev_torch.cpu().numpy(), "state": observation.state.cpu().numpy()}
            norm_data = self.data_transforms(tmp_data)
            a_prev_tensor = torch.from_numpy(norm_data["actions"]).to(self.device).float()
            # print("the a_prev_tensor is:", a_prev_tensor.shape)
            # print("the a_prev_tensor is:", a_prev_tensor)

        # 3. Run inference
        if a_prev_tensor is None:
            # First-time inference, use standard unguided sampling
            with torch.no_grad():
                action_norm = self.model.sample_actions(self.device, observation, num_steps=self.num_steps)
        else:
            # RTC guided sampling (gradients are needed internally, do not use torch.no_grad)
            # Assumes you have implemented guided_sample_actions in pi0_pytorch.py as previously suggested
            action_norm = self.model.guided_noise_sample_actions(
                self.device, 
                observation, 
                a_prev=a_prev_tensor, 
                d=d
            )

        action_norm = action_norm.cpu().numpy()
        predicted_action = action_norm[0,:self.chunk_size, :16]
        return predicted_action



    def predict_rtc_action(self, obs, a_prev=None, d=0, s=0, guidance_scale=1.0):
        """
        RTC-specific inference interface

        Args:
            obs: raw observation dict
            a_prev: raw action chunk generated at the previous timestep (numpy array)
            d: predicted number of inference latency steps
            s: number of steps the robot has executed since the last inference started
        """
        # 1. Preprocess the current observation
        observation = self.preprocess_observation(obs)

        # 2. Process the reference action chunk a_prev
        a_prev_tensor = None
        if a_prev is not None:
            # RTC guidance requires actions to be in normalized space
            # Convert numpy to tensor
            if isinstance(a_prev, list):
                a_prev = np.array(a_prev)
            a_prev_torch = torch.from_numpy(a_prev).to(self.device).float()

            # [Important] If a_prev is an unnormalized physical value, it must first be normalized back via the transform
            # Here we assume a_prev has shape [H, 16] or [1, H, 16]
            if a_prev_torch.ndim == 2:
                a_prev_torch = a_prev_torch.unsqueeze(0)

            # Use the existing transform to normalize (actions only)
            # Note: this requires that your NormalizeStatesActions supports handling actions separately
            # print("the a_prev_torch is:", a_prev_torch.shape)
            # print("the a_prev_torch is:", a_prev_torch)
            tmp_data = {"actions": a_prev_torch.cpu().numpy()}
            norm_data = self.data_transforms(tmp_data)
            a_prev_tensor = torch.from_numpy(norm_data["actions"]).to(self.device).float()
            # print("the a_prev_tensor is:", a_prev_tensor.shape)
            # print("the a_prev_tensor is:", a_prev_tensor)

        # 3. Run inference
        if a_prev_tensor is None:
            # First-time inference, use standard unguided sampling
            with torch.no_grad():
                action_norm = self.model.sample_actions(self.device, observation, num_steps=self.num_steps)
        else:
            # RTC guided sampling (gradients are needed internally, do not use torch.no_grad)
            # Assumes you have implemented guided_sample_actions in pi0_pytorch.py as previously suggested
            # import pdb; pdb.set_trace()
            action_norm = self.model.guided_sample_actions(
                self.device, 
                observation, 
                a_prev=a_prev_tensor, 
                d=d, 
                s=s, 
                num_steps=self.num_steps, 
                guidance_scale=guidance_scale
            )

        action_norm = action_norm.cpu().numpy()
        predicted_action = action_norm[0,:self.chunk_size, :16]
        return predicted_action



    def get_observation_features(self, obs):
        observation = self.preprocess_observation(obs)
        observation_features = self.model.get_observation_features(observation)
        return observation_features