import torch
from config.data_config import ONLINE_BUFFER_KEYS
import numpy as np
import os
# from openpi.models.hl_gauss_distribution import HLGaussDistribution
from algorithm.hl_gauss_distribution import HLGaussDistribution, DynamicHLGauss


class XHumanoidLerobotOfflineWrapper(torch.utils.data.Dataset):
    def __init__(self, offline_dataset, sample_from_ratio = 0.1, transforms = None, image_transform = None):
        self.offline_dataset = offline_dataset
        self.sample_from_ratio = sample_from_ratio
        origin_dataset_len = len(offline_dataset)
        self.sample_len = int(origin_dataset_len * sample_from_ratio)
        # Evenly spaced sampling instead of random sampling
        step = origin_dataset_len // self.sample_len if self.sample_len > 0 else 1
        self.sample_indices = list(range(0, origin_dataset_len, step))[:self.sample_len]
        self.transforms = transforms
        self.image_transform = image_transform

    def reset_sample_indices(self):
        # Evenly spaced sampling instead of random sampling
        step = len(self.offline_dataset) // self.sample_len if self.sample_len > 0 else 1
        self.sample_indices = list(range(0, len(self.offline_dataset), step))[:self.sample_len]

    def __len__(self):
        return self.sample_len

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        data = self.offline_dataset[sample_idx]
        align_data = {}
        # print("the data is:", data.keys())
        # for key in X_HUMANOID_DATA_SPEC.keys():
        #     if "image" in key:
        #         align_data[key] = data[key].permute(1,2,0)
        # print("the data['observation.image.image'].shape is:", data['observation.image.image'].shape)
        # cv2.imwrite("lerobot_observation.jpg", (data['observation.image.image'].permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8))
        align_data['observation.rgb_images.camera_front'] = data['observation.image.image'].permute(1,2,0)
        align_data['observation.rgb_images.camera_wrist'] = np.zeros_like(data['observation.image.image'].permute(1,2,0))
        
        
        left_gripper = data["observation.state.puppet_left_gripper_position"].unsqueeze(-1)
        right_gripper = data["observation.state.puppet_right_gripper_position"].unsqueeze(-1)
        master_left_gripper = data["observation.state.master_left_gripper_position"].unsqueeze(-1)
        master_right_gripper = data["observation.state.master_right_gripper_position"].unsqueeze(-1)
        state = torch.cat([
        data["observation.state.puppet_left_arm_position"], 
        left_gripper, data["observation.state.puppet_right_arm_position"],
        right_gripper], dim=-1)
        
        align_data['observation.state.arm_joint_position'] = state
        
        # print("observation.state.master_left_arm_position shape is:", data["observation.state.master_left_arm_position"].shape)
        # print("observation.state.master_right_arm_position shape is:", data["observation.state.master_right_arm_position"].shape)
        # print("observation.state.master_left_gripper_position shape is:", master_left_gripper.shape)
        # print("observation.state.master_right_gripper_position shape is:", master_right_gripper.shape)
        
        action = torch.cat([
        data["observation.state.master_left_arm_position"], 
        master_left_gripper, data["observation.state.master_right_arm_position"],
        master_right_gripper], dim=-1)
        
        align_data['action.arm_joint_position'] = action
        # reward = data["reward"]
        # align_data["reward"] = reward

        align_data['reward'] = data['reward']
        align_data['reward_mask'] = data['reward_mask']
        align_data['human_intervention'] = data['human_intervention']


        align_data['next_observation.rgb_images.camera_front'] = data['next_observation.image.image'].permute(1,2,0)
        align_data['next_observation.rgb_images.camera_wrist'] = np.zeros_like(data['next_observation.image.image'].permute(1,2,0))
        
        next_left_arm_position = data["next_observation.state.puppet_left_arm_position"]
        next_right_arm_position = data["next_observation.state.puppet_right_arm_position"]
        next_left_gripper = data["next_observation.state.puppet_left_gripper_position"].unsqueeze(-1)
        next_right_gripper = data["next_observation.state.puppet_right_gripper_position"].unsqueeze(-1)
        
        next_state = torch.cat([next_left_arm_position, next_left_gripper, next_right_arm_position, next_right_gripper], dim=-1)
        align_data['next_observation.state.arm_joint_position'] = next_state
        


        for key in ONLINE_BUFFER_KEYS:
            align_data[key] = data[key]
        
        if self.transforms is not None:
            data =  self.transforms(align_data)
            if self.image_transform is not None:
                for key in data.keys():
                    if 'image' in key and 'mask' not in key:
                        for sub_key in data[key].keys():
                            # print("the sub_key is:", sub_key)
                            # print("the data[key][sub_key] is:", data[key][sub_key].shape)
                            # import pdb; pdb.set_trace()
                            image = data[key][sub_key].transpose(1,2,0)
                            transformed_image = self.image_transform(image = image)['image']
                            data[key][sub_key] = transformed_image.transpose(2,0,1)
            return data
        else:   
            return align_data





def map_discounted_value_return_to_label(discounted_value_return, num_bin = 200):
    
    discounted_value_return = torch.clamp(discounted_value_return, -1, 0)
    label = (-discounted_value_return * num_bin).to(torch.int64).clamp(0, num_bin - 1)
    return label

import torch
import numpy as np
class XHumanoidValueFunctionLerobotOfflineWrapperForBC(torch.utils.data.Dataset):
    def __init__(self, offline_dataset, sample_from_ratio=1, transforms=None, image_transform=None):
        self.offline_dataset = offline_dataset
        self.sample_from_ratio = sample_from_ratio
        origin_dataset_len = len(offline_dataset)
        self.sample_len = int(origin_dataset_len * sample_from_ratio)
        # Evenly spaced sampling instead of random sampling
        step = origin_dataset_len // self.sample_len if self.sample_len > 0 else 1
        self.sample_indices = list(range(0, origin_dataset_len, step))[:self.sample_len]
        self.transforms = transforms
        self.image_transform = image_transform
        
        # Initialize distribution parameters consistent with the reference class
        num_value_bins = 512
        sigma = 3.0
        bins = torch.linspace(0, 400, num_value_bins)
        self.discounted_hl_gauss_distribution = HLGaussDistribution(bins, sigma)
        
        # Added: time distribution initialization
        time_bins = torch.linspace(0, 5000, num_value_bins)
        self.time_hl_gauss_distribution = HLGaussDistribution(time_bins, sigma)
    
    def reset_sample_indices(self):
        # Evenly spaced sampling instead of random sampling
        step = len(self.offline_dataset) // self.sample_len if self.sample_len > 0 else 1
        self.sample_indices = list(range(0, len(self.offline_dataset), step))[:self.sample_len]

    def __len__(self):
        return self.sample_len

    def _parse_image(self, image_data):
        """Image parsing logic from the reference code: convert (C,H,W) to (H,W,C) and to numpy"""
        if isinstance(image_data, torch.Tensor):
            # Assume LeRobot stores data as float32 (C, H, W)
            return image_data.permute(1, 2, 0).numpy()
        return image_data

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        data = self.offline_dataset[sample_idx]
        
        # for key in data.keys():
        #     if "image" in key:
        #         print("the key is:", key)
        #         print("the data[key] is:", data[key].shape)
        
        # --- 1. Image preprocessing (Multi-Camera) ---
        # Try to get images from three viewpoints, zero-fill if missing
        base_image = self._parse_image(data.get("observation.image.image", torch.zeros((3, 224, 224))))
        left_wrist_image = self._parse_image(data.get("observation.image.left", torch.zeros((3, 224, 224))))
        right_wrist_image = self._parse_image(data.get("observation.image.right", torch.zeros((3, 224, 224))))

        # --- 2. State and Action concatenation ---
        # Reference code logic: Arm Position (6/7D) + Gripper (1D)
        def concat_state(arm_key, gripper_key):
            arm_pos = data[arm_key]
            gripper_pos = data[gripper_key]
            if not len(gripper_pos.shape) == len(arm_pos.shape):
                gripper_pos = data[gripper_key].unsqueeze(-1)
            return torch.cat([arm_pos, gripper_pos], dim=-1)

        # Concatenate Puppet (current state)
        puppet_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        puppet_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        state = torch.cat([puppet_left, puppet_right], dim=-1).numpy()

        # Concatenate Master (in reference code, action = master_state)
        master_left = concat_state("observation.state.master_left_arm_position", "observation.state.master_left_gripper_position")
        master_right = concat_state("observation.state.master_right_arm_position", "observation.state.master_right_gripper_position")
        master_state = torch.cat([master_left, master_right], dim=-1).numpy()
        
        # The dataset's direct action could be used too, but the reference code forces action = master_state
        action = master_state

        # --- 3. Label and regression value processing (Value, RTG, Time) ---
        # Value GT (HL-Gauss)
        value_gt = None
        if "discounted_value_return" in data:
            v = data['discounted_value_return']
            value_gt = self.discounted_hl_gauss_distribution.encode(v).numpy()

        # Return-to-go (RTG)
        return_to_go_gt = None
        if 'return_to_go' in data:
            # Reference code logic: divide by 50 and convert to int32
            return_to_go_gt = np.array(data['return_to_go'] // 50, dtype=np.int32)

        # Remain Time (HL-Gauss)
        remain_time_gt = None
        if 'remain_time' in data:
            remain_time_gt = self.time_hl_gauss_distribution.encode(data['remain_time']).numpy()

        # Frame Index
        frame_index = None
        if 'frame_index' in data:
            frame_index = np.array(data['frame_index'], dtype=np.int32)

        # --- 4. Assemble align_data ---
        # Organize data in the nested structure of the reference code, for subsequent transforms processing
        align_data = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "actions": action,
            "value_gt": value_gt,
            "return_to_go_gt": return_to_go_gt,
            "remain_time_gt": remain_time_gt,
            "frame_index": frame_index,
        }

        # Include language instruction
        if "task" in data:
            align_data["prompt"] = data["task"]
        
        # Compatibility with other keys that older code may need
        if "action_judgement" in data:
            align_data["action_judgement"] = data["action_judgement"].numpy()

        # --- 5. Apply Transforms ---
        if self.transforms is not None:
            # The transforms here should be X_Humanoid_hl_gauss_RTG_Multi_Camera_Inputs 
            # or a similar transform for the dict structure
            transformed_data = self.transforms(align_data)
            
            
            # If there's additional image augmentation (e.g. RandomColorJitter)
            if self.image_transform is not None:
                for key in ['image']:
                    if key in transformed_data:
                        for sub_key in transformed_data[key].keys():
                            img = transformed_data[key][sub_key]
                            # Note: A.Compose usually expects uint8 HWC format
                            transformed_data[key][sub_key] = self.image_transform(image=img)['image']
            return transformed_data
        
        return align_data




class XHumanoidValueFunctionWithPreHistoryLerobotOfflineWrapperForBCWithOnlyClassification(torch.utils.data.Dataset):
    def __init__(self, offline_dataset, sample_from_ratio=1, transforms=None, image_transform=None, use_hl_gauss=True):
        self.offline_dataset = offline_dataset
        self.sample_from_ratio = sample_from_ratio
        origin_dataset_len = len(offline_dataset)
        self.sample_len = int(origin_dataset_len * sample_from_ratio)
        # Evenly spaced sampling instead of random sampling
        step = origin_dataset_len // self.sample_len if self.sample_len > 0 else 1
        self.sample_indices = list(range(0, origin_dataset_len, step))[:self.sample_len]
        self.transforms = transforms
        self.image_transform = image_transform
        
        # Initialize distribution parameters consistent with the reference class
        num_value_bins = 256
        self.num_value_bins = num_value_bins
        sigma = 3.0
        bins = torch.linspace(0, 400, num_value_bins)
        self.discounted_hl_gauss_distribution = HLGaussDistribution(bins, sigma)
        
        # Added: time distribution initialization
        # time_bins = torch.linspace(0, 5000, num_value_bins)
        # self.time_hl_gauss_distribution = HLGaussDistribution(time_bins, sigma)
        self.max_time = 5000
        self.skip_time = self.max_time // num_value_bins
    
    def reset_sample_indices(self):
        # Evenly spaced sampling instead of random sampling
        step = len(self.offline_dataset) // self.sample_len if self.sample_len > 0 else 1
        self.sample_indices = list(range(0, len(self.offline_dataset), step))[:self.sample_len]

    def __len__(self):
        return self.sample_len

    def _parse_image(self, image_data):
        """Image parsing logic from the reference code: convert (C,H,W) to (H,W,C) and to numpy"""
        if isinstance(image_data, torch.Tensor):
            # Assume LeRobot stores data as float32 (C, H, W)
            return image_data.permute(1, 2, 0).numpy()
        return image_data

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        data = self.offline_dataset[sample_idx]
        
        # for key in data.keys():
        #     if "image" in key:
        #         print("the key is:", key)
        #         print("the data[key] is:", data[key].shape)
        
        # --- 1. Image preprocessing (Multi-Camera) ---
        # Try to get images from three viewpoints, zero-fill if missing
        # print("the observation.image.left shape is:", data.get("observation.image.left").shape)
        # print("the observation.image.right shape is:", data.get("observation.image.right").shape)
        # print("the observation.image.image shape is:", data.get("observation.image.image").shape)
        pre_history_image = data.get("observation.image.image")[0]
        base_image = data.get("observation.image.image")[1]
        
        pre_history_image = self._parse_image(pre_history_image)
        base_image = self._parse_image(base_image)
        # base_image = self._parse_image(data.get("observation.image.image", torch.zeros((3, 224, 224))))
        left_wrist_image = self._parse_image(data.get("observation.image.left", torch.zeros((3, 224, 224))))
        right_wrist_image = self._parse_image(data.get("observation.image.right", torch.zeros((3, 224, 224))))

        # print("the pre_history_image shape is:", pre_history_image.shape)
        # print("the base_image shape is:", base_image.shape)

        # --- 2. State and Action concatenation ---
        # Reference code logic: Arm Position (6/7D) + Gripper (1D)
        def concat_state(arm_key, gripper_key):
            arm_pos = data[arm_key]
            gripper_pos = data[gripper_key]
            if not len(gripper_pos.shape) == len(arm_pos.shape):
                gripper_pos = data[gripper_key].unsqueeze(-1)
            return torch.cat([arm_pos, gripper_pos], dim=-1)

        # Concatenate Puppet (current state)
        puppet_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        puppet_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        state = torch.cat([puppet_left, puppet_right], dim=-1).numpy()

        # Concatenate Master (in reference code, action = master_state)
        master_left = concat_state("observation.state.master_left_arm_position", "observation.state.master_left_gripper_position")
        master_right = concat_state("observation.state.master_right_arm_position", "observation.state.master_right_gripper_position")
        master_state = torch.cat([master_left, master_right], dim=-1).numpy()
        
        # The dataset's direct action could be used too, but the reference code forces action = master_state
        action = master_state

        # --- 3. Label and regression value processing (Value, RTG, Time) ---
        # Value GT (HL-Gauss)
        value_gt = None
        if "discounted_value_return" in data:
            v = data['discounted_value_return']
            value_gt = self.discounted_hl_gauss_distribution.encode(v).numpy()

        # Return-to-go (RTG)
        return_to_go_gt = None
        changed_return_to_go = None
        if 'return_to_go' in data:
            # Reference code logic: divide by 50 and convert to int32
            if isinstance(data['return_to_go'], torch.Tensor):
                return_to_go_gt = data['return_to_go'][0].item() // 50
                return_to_go_gt = np.array(return_to_go_gt, dtype=np.int32)
                return_to_go_gt = np.clip(return_to_go_gt, 0, 10)
                changed_return_to_go = int(data['return_to_go'][0].item() != data['return_to_go'][-1].item())
                changed_return_to_go = np.array(changed_return_to_go, dtype=np.int32)
            else:
                return_to_go_gt = np.array(data['return_to_go'][0] // 50, dtype=np.int32)
                changed_return_to_go = np.array(int(data['return_to_go'][0] != data['return_to_go'][-1]), dtype=np.int32)
        
        

        # Remain Time (one-hot)
        remain_time_gt = None
        if 'remain_time' in data:
            remain_time_idx = np.array(data['remain_time'], dtype=np.int32)
            remain_time_idx = remain_time_idx // self.skip_time
            remain_time_idx = np.clip(remain_time_idx, 0, self.num_value_bins - 1)
            remain_time_gt = np.zeros(self.num_value_bins, dtype=np.float32)
            remain_time_gt[remain_time_idx] = 1.0

        # Frame Index
        frame_index = None
        if 'frame_index' in data:
            frame_index = np.array(data['frame_index'], dtype=np.int32)

        # --- 4. Assemble align_data ---
        # Organize data in the nested structure of the reference code, for subsequent transforms processing
        align_data = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "history_image": pre_history_image,
            "actions": action,
            
            "value_gt": value_gt,
            "return_to_go_gt": return_to_go_gt,
            "remain_time_gt": remain_time_gt,
            "frame_index": frame_index,
            "changed_return_to_go": changed_return_to_go
        }
        

        # Include language instruction
        if "task" in data:
            align_data["prompt"] = data["task"]
        
        # Compatibility with other keys that older code may need
        if "action_judgement" in data:
            align_data["action_judgement"] = data["action_judgement"].numpy()

        # --- 5. Apply Transforms ---
        if self.transforms is not None:
            # The transforms here should be X_Humanoid_hl_gauss_RTG_Multi_Camera_Inputs 
            # or a similar transform for the dict structure
            transformed_data = self.transforms(align_data)
            
            # print("the transformed_data is:", transformed_data.keys())
            
            # If there's additional image augmentation (e.g. RandomColorJitter)
            if self.image_transform is not None:
                for key in ['image']:
                    if key in transformed_data:
                        for sub_key in transformed_data[key].keys():
                            img = transformed_data[key][sub_key]
                            # Note: A.Compose usually expects uint8 HWC format
                            transformed_data[key][sub_key] = self.image_transform(image=img)['image']
                if 'history_image' in transformed_data:
                    img = transformed_data['history_image']
                    transformed_data['history_image'] = self.image_transform(image=img)['image']
            return transformed_data

        return align_data
    


class XHumanoidValueFunctionWithPreHistoryLerobotOfflineWrapperForBC(torch.utils.data.Dataset):
    def __init__(self, offline_dataset, sample_from_ratio=1, transforms=None, image_transform=None, use_hl_gauss=True):
        self.offline_dataset = offline_dataset
        self.sample_from_ratio = sample_from_ratio
        origin_dataset_len = len(offline_dataset)
        self.sample_len = int(origin_dataset_len * sample_from_ratio)
        # Evenly spaced sampling instead of random sampling
        step = origin_dataset_len // self.sample_len if self.sample_len > 0 else 1
        self.sample_indices = list(range(0, origin_dataset_len, step))[:self.sample_len]
        self.transforms = transforms
        self.image_transform = image_transform
        
        # Initialize distribution parameters consistent with the reference class
        num_value_bins = 256
        sigma = 3.0
        bins = torch.linspace(0, 400, num_value_bins)
        self.discounted_hl_gauss_distribution = HLGaussDistribution(bins, sigma)
        
        # Added: time distribution initialization
        time_bins = torch.linspace(0, 5000, num_value_bins)
        self.time_hl_gauss_distribution = HLGaussDistribution(time_bins, sigma)
    
    def reset_sample_indices(self):
        # Evenly spaced sampling instead of random sampling
        step = len(self.offline_dataset) // self.sample_len if self.sample_len > 0 else 1
        self.sample_indices = list(range(0, len(self.offline_dataset), step))[:self.sample_len]

    def __len__(self):
        return self.sample_len

    def _parse_image(self, image_data):
        """Image parsing logic from the reference code: convert (C,H,W) to (H,W,C) and to numpy"""
        if isinstance(image_data, torch.Tensor):
            # Assume LeRobot stores data as float32 (C, H, W)
            return image_data.permute(1, 2, 0).numpy()
        return image_data

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        data = self.offline_dataset[sample_idx]
        
        # for key in data.keys():
        #     if "image" in key:
        #         print("the key is:", key)
        #         print("the data[key] is:", data[key].shape)
        
        # --- 1. Image preprocessing (Multi-Camera) ---
        # Try to get images from three viewpoints, zero-fill if missing
        # print("the observation.image.left shape is:", data.get("observation.image.left").shape)
        # print("the observation.image.right shape is:", data.get("observation.image.right").shape)
        # print("the observation.image.image shape is:", data.get("observation.image.image").shape)
        pre_history_image = data.get("observation.image.image")[0]
        base_image = data.get("observation.image.image")[1]
        
        pre_history_image = self._parse_image(pre_history_image)
        base_image = self._parse_image(base_image)
        # base_image = self._parse_image(data.get("observation.image.image", torch.zeros((3, 224, 224))))
        left_wrist_image = self._parse_image(data.get("observation.image.left", torch.zeros((3, 224, 224))))
        right_wrist_image = self._parse_image(data.get("observation.image.right", torch.zeros((3, 224, 224))))

        # print("the pre_history_image shape is:", pre_history_image.shape)
        # print("the base_image shape is:", base_image.shape)

        # --- 2. State and Action concatenation ---
        # Reference code logic: Arm Position (6/7D) + Gripper (1D)
        def concat_state(arm_key, gripper_key):
            arm_pos = data[arm_key]
            gripper_pos = data[gripper_key]
            if not len(gripper_pos.shape) == len(arm_pos.shape):
                gripper_pos = data[gripper_key].unsqueeze(-1)
            return torch.cat([arm_pos, gripper_pos], dim=-1)

        # Concatenate Puppet (current state)
        puppet_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        puppet_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        state = torch.cat([puppet_left, puppet_right], dim=-1).numpy()

        # Concatenate Master (in reference code, action = master_state)
        master_left = concat_state("observation.state.master_left_arm_position", "observation.state.master_left_gripper_position")
        master_right = concat_state("observation.state.master_right_arm_position", "observation.state.master_right_gripper_position")
        master_state = torch.cat([master_left, master_right], dim=-1).numpy()
        
        # The dataset's direct action could be used too, but the reference code forces action = master_state
        action = master_state

        # --- 3. Label and regression value processing (Value, RTG, Time) ---
        # Value GT (HL-Gauss)
        value_gt = None
        if "discounted_value_return" in data:
            v = data['discounted_value_return']
            value_gt = self.discounted_hl_gauss_distribution.encode(v).numpy()

        # Return-to-go (RTG)
        return_to_go_gt = None
        changed_return_to_go = None
        if 'return_to_go' in data:
            # Reference code logic: divide by 50 and convert to int32
            if isinstance(data['return_to_go'], torch.Tensor):
                return_to_go_gt = data['return_to_go'][0].item() // 50
                return_to_go_gt = np.array(return_to_go_gt, dtype=np.int32)
                return_to_go_gt = np.clip(return_to_go_gt, 0, 10)
                changed_return_to_go = int(data['return_to_go'][0].item() != data['return_to_go'][-1].item())
                changed_return_to_go = np.array(changed_return_to_go, dtype=np.int32)
            else:
                return_to_go_gt = np.array(data['return_to_go'][0] // 50, dtype=np.int32)
                changed_return_to_go = np.array(int(data['return_to_go'][0] != data['return_to_go'][-1]), dtype=np.int32)
        
        

        # Remain Time (HL-Gauss)
        remain_time_gt = None
        if 'remain_time' in data:
            remain_time = data['remain_time']
            remain_time = np.clip(remain_time, 0, 5000)
            remain_time_gt = self.time_hl_gauss_distribution.encode(remain_time).numpy()

        # Frame Index
        frame_index = None
        if 'frame_index' in data:
            frame_index = np.array(data['frame_index'], dtype=np.int32)

        # --- 4. Assemble align_data ---
        # Organize data in the nested structure of the reference code, for subsequent transforms processing
        align_data = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "history_image": pre_history_image,
            "actions": action,
            
            "value_gt": value_gt,
            "return_to_go_gt": return_to_go_gt,
            "remain_time_gt": remain_time_gt,
            "frame_index": frame_index,
            "changed_return_to_go": changed_return_to_go
        }
        

        # Include language instruction
        if "task" in data:
            align_data["prompt"] = data["task"]
        
        # Compatibility with other keys that older code may need
        if "action_judgement" in data:
            align_data["action_judgement"] = data["action_judgement"].numpy()

        # --- 5. Apply Transforms ---
        if self.transforms is not None:
            # The transforms here should be X_Humanoid_hl_gauss_RTG_Multi_Camera_Inputs 
            # or a similar transform for the dict structure
            transformed_data = self.transforms(align_data)
            
            # print("the transformed_data is:", transformed_data.keys())
            
            # If there's additional image augmentation (e.g. RandomColorJitter)
            if self.image_transform is not None:
                for key in ['image']:
                    if key in transformed_data:
                        for sub_key in transformed_data[key].keys():
                            img = transformed_data[key][sub_key]
                            # Note: A.Compose usually expects uint8 HWC format
                            transformed_data[key][sub_key] = self.image_transform(image=img)['image']
                if 'history_image' in transformed_data:
                    img = transformed_data['history_image']
                    transformed_data['history_image'] = self.image_transform(image=img)['image']
            return transformed_data

        return align_data


class XHumanoidValueFunctionLerobotOfflineWrapperForBCWithDynamicHLGauss(torch.utils.data.Dataset):
    """
    Wrapper that uses DynamicHLGauss for data labeling
    DynamicHLGauss uses a dynamic sigma (based on local bin spacing), which adapts better to the data distribution than a fixed sigma
    """
    def __init__(
        self,
        offline_dataset,
        discounted_value_bins_path=None,
        remain_time_bins_path=None,
        num_value_bins=256,
        sigma_scale=0.75,
        sample_from_ratio=1,
        transforms=None,
        image_transform=None
    ):
        """
        Args:
            offline_dataset: offline dataset
            discounted_value_bins_path: path to precomputed discounted_value_return bins (.npy)
            remain_time_bins_path: path to precomputed remain_time bins (.npy)
            num_value_bins: number of bins, default 256
            sigma_scale: dynamic sigma scaling factor, default 0.75
            sample_from_ratio: sampling ratio
            transforms: data transform
            image_transform: image transform
        """
        self.offline_dataset = offline_dataset
        self.sample_from_ratio = sample_from_ratio
        origin_dataset_len = len(offline_dataset)
        self.sample_len = int(origin_dataset_len * sample_from_ratio)
        # Evenly spaced sampling instead of random sampling
        step = origin_dataset_len // self.sample_len if self.sample_len > 0 else 1
        self.sample_indices = list(range(0, origin_dataset_len, step))[:self.sample_len]
        self.transforms = transforms
        self.image_transform = image_transform

        # Load or create the DynamicHLGauss for discounted_value_return
        if discounted_value_bins_path is not None and os.path.exists(discounted_value_bins_path):
            print(f"Loading discounted_value_return bins from: {discounted_value_bins_path}")
            discount_value_bins = torch.from_numpy(np.load(discounted_value_bins_path))
            self.discounted_hl_gauss_distribution = DynamicHLGauss(bins=discount_value_bins, sigma_scale=sigma_scale)
        else:
            print("Warning: discounted_value_bins_path not provided, using default uniform distribution")
            default_bins = torch.linspace(0, 400, num_value_bins)
            self.discounted_hl_gauss_distribution = DynamicHLGauss(bins=default_bins, sigma_scale=sigma_scale)

        # Load or create the DynamicHLGauss for remain_time
        if remain_time_bins_path is not None and os.path.exists(remain_time_bins_path):
            print(f"Loading remain_time bins from: {remain_time_bins_path}")
            time_bins = torch.from_numpy(np.load(remain_time_bins_path))
            self.time_hl_gauss_distribution = DynamicHLGauss(bins=time_bins, sigma_scale=sigma_scale)
        else:
            print("Warning: remain_time_bins_path not provided, using default uniform distribution")
            default_time_bins = torch.linspace(0, 5000, num_value_bins)
            self.time_hl_gauss_distribution = DynamicHLGauss(bins=default_time_bins, sigma_scale=sigma_scale)

    def reset_sample_indices(self):
        # Evenly spaced sampling instead of random sampling
        step = len(self.offline_dataset) // self.sample_len if self.sample_len > 0 else 1
        self.sample_indices = list(range(0, len(self.offline_dataset), step))[:self.sample_len]

    def __len__(self):
        return self.sample_len

    def _parse_image(self, image_data):
        """Convert (C,H,W) to (H,W,C) and to numpy"""
        if isinstance(image_data, torch.Tensor):
            return image_data.permute(1, 2, 0).numpy()
        return image_data

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        data = self.offline_dataset[sample_idx]

        # --- 1. Image preprocessing (Multi-Camera) ---
        base_image = self._parse_image(data.get("observation.image.image", torch.zeros((3, 224, 224))))
        left_wrist_image = self._parse_image(data.get("observation.image.left", torch.zeros((3, 224, 224))))
        right_wrist_image = self._parse_image(data.get("observation.image.right", torch.zeros((3, 224, 224))))

        # --- 2. State and Action concatenation ---
        def concat_state(arm_key, gripper_key):
            arm_pos = data[arm_key]
            gripper_pos = data[gripper_key]
            if not len(gripper_pos.shape) == len(arm_pos.shape):
                gripper_pos = data[gripper_key].unsqueeze(-1)
            return torch.cat([arm_pos, gripper_pos], dim=-1)

        # Concatenate Puppet (current state)
        puppet_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        puppet_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        state = torch.cat([puppet_left, puppet_right], dim=-1).numpy()

        # Concatenate Master (action = master_state)
        master_left = concat_state("observation.state.master_left_arm_position", "observation.state.master_left_gripper_position")
        master_right = concat_state("observation.state.master_right_arm_position", "observation.state.master_right_gripper_position")
        master_state = torch.cat([master_left, master_right], dim=-1).numpy()
        action = master_state

        # --- 3. Label and regression value processing (using DynamicHLGauss) ---
        # Value GT (using DynamicHLGauss)
        value_gt = None
        if "discounted_value_return" in data:
            v = data['discounted_value_return']
            # DynamicHLGauss.encode returns a Tensor, needs to be converted to numpy
            value_gt = self.discounted_hl_gauss_distribution.encode(v).cpu().numpy()

        # Return-to-go (RTG)
        return_to_go_gt = None
        if 'return_to_go' in data:
            if isinstance(data['return_to_go'], torch.Tensor):
                return_to_go_gt = np.array(data['return_to_go'].item() // 50, dtype=np.int32)
            else:
                return_to_go_gt = np.array(data['return_to_go'] // 50, dtype=np.int32)

        # Remain Time (using DynamicHLGauss)
        remain_time_gt = None
        if 'remain_time' in data:
            remain_time_gt = self.time_hl_gauss_distribution.encode(data['remain_time']).cpu().numpy()

        # Frame Index
        frame_index = None
        if 'frame_index' in data:
            frame_index = np.array(data['frame_index'], dtype=np.int32)

        # --- 4. Assemble align_data ---
        align_data = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "actions": action,
            "value_gt": value_gt,
            "return_to_go_gt": return_to_go_gt,
            "remain_time_gt": remain_time_gt,
            "frame_index": frame_index,
        }

        # Include language instruction
        if "task" in data:
            align_data["prompt"] = data["task"]

        # Compatibility with other keys that older code may need
        if "action_judgement" in data:
            align_data["action_judgement"] = data["action_judgement"].numpy()

        # --- 5. Apply Transforms ---
        if self.transforms is not None:
            transformed_data = self.transforms(align_data)

            # If there's additional image augmentation
            if self.image_transform is not None:
                for key in ['image']:
                    if key in transformed_data:
                        for sub_key in transformed_data[key].keys():
                            img = transformed_data[key][sub_key]
                            transformed_data[key][sub_key] = self.image_transform(image=img)['image']
            return transformed_data

        return align_data


class XHumanoidValueFunctionWithPreHistoryLerobotOfflineWrapperForBCWithDynamicHLGauss(torch.utils.data.Dataset):
    """
    Wrapper that uses DynamicHLGauss for data labeling (with Pre-History image version)
    """
    def __init__(
        self,
        offline_dataset,
        discounted_value_bins_path=None,
        remain_time_bins_path=None,
        num_value_bins=256,
        sigma_scale=0.75,
        sample_from_ratio=1,
        transforms=None,
        image_transform=None
    ):
        """
        Args:
            offline_dataset: offline dataset
            discounted_value_bins_path: path to precomputed discounted_value_return bins (.npy)
            remain_time_bins_path: path to precomputed remain_time bins (.npy)
            num_value_bins: number of bins, default 256
            sigma_scale: dynamic sigma scaling factor, default 0.75
            sample_from_ratio: sampling ratio
            transforms: data transform
            image_transform: image transform
        """
        self.offline_dataset = offline_dataset
        self.sample_from_ratio = sample_from_ratio
        origin_dataset_len = len(offline_dataset)
        self.sample_len = int(origin_dataset_len * sample_from_ratio)
        # Evenly spaced sampling instead of random sampling
        step = origin_dataset_len // self.sample_len if self.sample_len > 0 else 1
        self.sample_indices = list(range(0, origin_dataset_len, step))[:self.sample_len]
        self.transforms = transforms
        self.image_transform = image_transform

        # Load or create the DynamicHLGauss for discounted_value_return
        if discounted_value_bins_path is not None and os.path.exists(discounted_value_bins_path):
            print(f"Loading discounted_value_return bins from: {discounted_value_bins_path}")
            discount_value_bins = torch.from_numpy(np.load(discounted_value_bins_path))
            self.discounted_hl_gauss_distribution = DynamicHLGauss(bins=discount_value_bins, sigma_scale=sigma_scale)
        else:
            print("Warning: discounted_value_bins_path not provided, using default uniform distribution")
            default_bins = torch.linspace(-600, 600, num_value_bins)
            self.discounted_hl_gauss_distribution = DynamicHLGauss(bins=default_bins, sigma_scale=sigma_scale)

        # Load or create the DynamicHLGauss for remain_time
        if remain_time_bins_path is not None and os.path.exists(remain_time_bins_path):
            print(f"Loading remain_time bins from: {remain_time_bins_path}")
            time_bins = torch.from_numpy(np.load(remain_time_bins_path))
            self.time_hl_gauss_distribution = DynamicHLGauss(bins=time_bins, sigma_scale=sigma_scale)
        else:
            print("Warning: remain_time_bins_path not provided, using default uniform distribution")
            default_time_bins = torch.linspace(0, 5000, num_value_bins)
            self.time_hl_gauss_distribution = DynamicHLGauss(bins=default_time_bins, sigma_scale=sigma_scale)

    def reset_sample_indices(self):
        # Evenly spaced sampling instead of random sampling
        step = len(self.offline_dataset) // self.sample_len if self.sample_len > 0 else 1
        self.sample_indices = list(range(0, len(self.offline_dataset), step))[:self.sample_len]

    def __len__(self):
        return self.sample_len

    def _parse_image(self, image_data):
        """Convert (C,H,W) to (H,W,C) and to numpy"""
        if isinstance(image_data, torch.Tensor):
            return image_data.permute(1, 2, 0).numpy()
        return image_data

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        data = self.offline_dataset[sample_idx]

        # --- 1. Image preprocessing (with Pre-History) ---
        pre_history_image = data.get("observation.image.image")[0]
        base_image = data.get("observation.image.image")[1]

        pre_history_image = self._parse_image(pre_history_image)
        base_image = self._parse_image(base_image)
        left_wrist_image = self._parse_image(data.get("observation.image.left", torch.zeros((3, 224, 224))))
        right_wrist_image = self._parse_image(data.get("observation.image.right", torch.zeros((3, 224, 224))))

        # --- 2. State and Action concatenation ---
        def concat_state(arm_key, gripper_key):
            arm_pos = data[arm_key]
            gripper_pos = data[gripper_key]
            if not len(gripper_pos.shape) == len(arm_pos.shape):
                gripper_pos = data[gripper_key].unsqueeze(-1)
            return torch.cat([arm_pos, gripper_pos], dim=-1)

        # Concatenate Puppet (current state)
        puppet_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        puppet_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        state = torch.cat([puppet_left, puppet_right], dim=-1).numpy()

        # Concatenate Master (action = master_state)
        master_left = concat_state("observation.state.master_left_arm_position", "observation.state.master_left_gripper_position")
        master_right = concat_state("observation.state.master_right_arm_position", "observation.state.master_right_gripper_position")
        master_state = torch.cat([master_left, master_right], dim=-1).numpy()
        action = master_state

        # --- 3. Label and regression value processing (using DynamicHLGauss) ---
        # Value GT (using DynamicHLGauss)
        value_gt = None
        if "discounted_value_return" in data:
            v = data['discounted_value_return']
            value_gt = self.discounted_hl_gauss_distribution.encode(v).cpu().numpy()

        # Return-to-go (RTG)
        return_to_go_gt = None
        changed_return_to_go = None
        if 'return_to_go' in data:
            if isinstance(data['return_to_go'], torch.Tensor):
                return_to_go_gt = data['return_to_go'][0].item() // 50
                return_to_go_gt = np.array(return_to_go_gt, dtype=np.int32)
                return_to_go_gt = np.clip(return_to_go_gt, 0, 10)
                changed_return_to_go = int(data['return_to_go'][0].item() != data['return_to_go'][-1].item())
                changed_return_to_go = np.array(changed_return_to_go, dtype=np.int32)
            else:
                return_to_go_gt = np.array(data['return_to_go'][0] // 50, dtype=np.int32)
                changed_return_to_go = np.array(int(data['return_to_go'][0] != data['return_to_go'][-1]), dtype=np.int32)

        # Remain Time (using DynamicHLGauss)
        remain_time_gt = None
        if 'remain_time' in data:
            remain_time_gt = self.time_hl_gauss_distribution.encode(data['remain_time']).cpu().numpy()

        # Frame Index
        frame_index = None
        if 'frame_index' in data:
            frame_index = np.array(data['frame_index'], dtype=np.int32)

        # --- 4. Assemble align_data ---
        align_data = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "history_image": pre_history_image,
            "actions": action,
            "value_gt": value_gt,
            "return_to_go_gt": return_to_go_gt,
            "remain_time_gt": remain_time_gt,
            "frame_index": frame_index,
            "changed_return_to_go": changed_return_to_go
        }

        # Include language instruction
        if "task" in data:
            align_data["prompt"] = data["task"]

        # Compatibility with other keys that older code may need
        if "action_judgement" in data:
            align_data["action_judgement"] = data["action_judgement"].numpy()

        # --- 5. Apply Transforms ---
        if self.transforms is not None:
            transformed_data = self.transforms(align_data)

            # If there's additional image augmentation
            if self.image_transform is not None:
                for key in ['image']:
                    if key in transformed_data:
                        for sub_key in transformed_data[key].keys():
                            img = transformed_data[key][sub_key]
                            transformed_data[key][sub_key] = self.image_transform(image=img)['image']
                if 'history_image' in transformed_data:
                    img = transformed_data['history_image']
                    transformed_data['history_image'] = self.image_transform(image=img)['image']
            return transformed_data

        return align_data
