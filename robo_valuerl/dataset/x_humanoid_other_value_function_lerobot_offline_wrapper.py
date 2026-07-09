import torch
import numpy as np
import os
# from openpi.models.hl_gauss_distribution import HLGaussDistribution
from algorithm.hl_gauss_distribution import HLGaussDistribution, DynamicHLGauss


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


class OtherOpenXValueFunctionWithPreHistoryLerobotOfflineWrapperForBC(torch.utils.data.Dataset):
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
        pre_history_image = data.get("observation.images.image")[0]
        base_image = data.get("observation.images.image")[1]
        
        pre_history_image = self._parse_image(pre_history_image)
        base_image = self._parse_image(base_image)
        # base_image = self._parse_image(data.get("observation.image.image", torch.zeros((3, 224, 224))))
        left_wrist_image = self._parse_image(data.get("observation.images.wrist_image", torch.zeros((3, 224, 224))))
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
        # puppet_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        # puppet_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        # state = torch.cat([puppet_left, puppet_right], dim=-1).numpy()

        # Concatenate Master (in reference code, action = master_state)
        # master_left = concat_state("observation.state.master_left_arm_position", "observation.state.master_left_gripper_position")
        # master_right = concat_state("observation.state.master_right_arm_position", "observation.state.master_right_gripper_position")
        # master_state = torch.cat([master_left, master_right], dim=-1).numpy()
        
        # The dataset's direct action could be used too, but the reference code forces action = master_state
        # action = master_state

        state_joint_position = data['observation.state']
        action_joint_position = data['action']
        # Assume action_joint_position has shape [Batch, 10]
        # Create a [Batch, 2] all-zero Tensor to concatenate at the end
        action_zeros_to_add = torch.zeros_like(action_joint_position[..., :2]) 
        # print("the action_zeros_to_add shape is:", action_zeros_to_add.shape)
        # print("the action_joint_position shape is:", action_joint_position.shape)
        # print("the state_joint_position shape is:", state_joint_position.shape)
        state_zeros_to_add = torch.zeros_like(state_joint_position[..., :2]) 
        action = torch.cat([action_joint_position, action_zeros_to_add], dim=-1).numpy()
        state = torch.cat([state_joint_position, state_zeros_to_add], dim=-1).numpy()

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
                "left_wrist_0_rgb": np.True_ if "observation.images.wrist_image" in data.keys() else np.False_,
                "right_wrist_0_rgb": np.False_,
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



class OtherOpenXValueFunctionWithPreHistoryLerobotOfflineWrapperForBCWithOnlyClassification(torch.utils.data.Dataset):
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
        # print("the data key is:", data.keys())
        # print("the task is:", data.get("task"))
        # for key in data.keys():
        #     if "image" in key:
        #         print("the key is:", key)
        #         print("the data[key] is:", data[key].shape)
        
        # --- 1. Image preprocessing (Multi-Camera) ---
        # Try to get images from three viewpoints, zero-fill if missing
        # print("the observation.image.left shape is:", data.get("observation.image.left").shape)
        # print("the observation.image.right shape is:", data.get("observation.image.right").shape)
        # print("the observation.image.image shape is:", data.get("observation.image.image").shape)
        pre_history_image = data.get("observation.images.image")[0]
        base_image = data.get("observation.images.image")[1]
        
        pre_history_image = self._parse_image(pre_history_image)
        base_image = self._parse_image(base_image)
        # base_image = self._parse_image(data.get("observation.image.image", torch.zeros((3, 224, 224))))
        left_wrist_image = self._parse_image(data.get("observation.images.wrist_image", torch.zeros((3, 224, 224))))
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
        # puppet_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        # puppet_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        # state = torch.cat([puppet_left, puppet_right], dim=-1).numpy()

        # Concatenate Master (in reference code, action = master_state)
        # master_left = concat_state("observation.state.master_left_arm_position", "observation.state.master_left_gripper_position")
        # master_right = concat_state("observation.state.master_right_arm_position", "observation.state.master_right_gripper_position")
        # master_state = torch.cat([master_left, master_right], dim=-1).numpy()
        
        # The dataset's direct action could be used too, but the reference code forces action = master_state
        # action = master_state

        state_joint_position = data['observation.state']
        action_joint_position = data['action']
        # Assume action_joint_position has shape [Batch, 10]
        # Create a [Batch, 2] all-zero Tensor to concatenate at the end
        action_zeros_to_add = torch.zeros_like(action_joint_position[..., :2]) 
        # print("the action_zeros_to_add shape is:", action_zeros_to_add.shape)
        # print("the action_joint_position shape is:", action_joint_position.shape)
        # print("the state_joint_position shape is:", state_joint_position.shape)
        state_zeros_to_add = torch.zeros_like(state_joint_position[..., :2]) 
        action = torch.cat([action_joint_position, action_zeros_to_add], dim=-1).numpy()
        state = torch.cat([state_joint_position, state_zeros_to_add], dim=-1).numpy()

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
                "left_wrist_0_rgb": np.True_ if "observation.images.wrist_image" in data.keys() else np.False_,
                "right_wrist_0_rgb": np.False_,
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
        # print("the data prompt is:", data.get("prompt"))
        # print("the openx index is:", idx)
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

class OtherXHumanoidValueFunctionWithPreHistoryLerobotOfflineWrapperForBCWithOnlyClassification(torch.utils.data.Dataset):
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
        pre_history_image = data.get("observation.rgb_images.camera_head")[0]
        base_image = data.get("observation.rgb_images.camera_head")[1]
        
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
        # puppet_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        # puppet_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        # state = torch.cat([puppet_left, puppet_right], dim=-1).numpy()

        # Concatenate Master (in reference code, action = master_state)
        # master_left = concat_state("observation.state.master_left_arm_position", "observation.state.master_left_gripper_position")
        # master_right = concat_state("observation.state.master_right_arm_position", "observation.state.master_right_gripper_position")
        # master_state = torch.cat([master_left, master_right], dim=-1).numpy()
        
        # The dataset's direct action could be used too, but the reference code forces action = master_state
        # action = master_state

        state_joint_position = data['observation.state.arm_joint_position']
        action_joint_position = data['action.arm_joint_position']
        # Assume action_joint_position has shape [Batch, 10]
        # Create a [Batch, 2] all-zero Tensor to concatenate at the end
        action_zeros_to_add = torch.zeros_like(action_joint_position[..., :2]) 
        # print("the action_zeros_to_add shape is:", action_zeros_to_add.shape)
        # print("the action_joint_position shape is:", action_joint_position.shape)
        # print("the state_joint_position shape is:", state_joint_position.shape)
        state_zeros_to_add = torch.zeros_like(state_joint_position[..., :2]) 
        action = torch.cat([action_joint_position, action_zeros_to_add], dim=-1).numpy()
        state = torch.cat([state_joint_position, state_zeros_to_add], dim=-1).numpy()

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

class OtherXHumanoidValueFunctionWithPreHistoryLerobotOfflineWrapperForBC(torch.utils.data.Dataset):
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
        pre_history_image = data.get("observation.rgb_images.camera_head")[0]
        base_image = data.get("observation.rgb_images.camera_head")[1]
        
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
        # puppet_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        # puppet_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        # state = torch.cat([puppet_left, puppet_right], dim=-1).numpy()

        # Concatenate Master (in reference code, action = master_state)
        # master_left = concat_state("observation.state.master_left_arm_position", "observation.state.master_left_gripper_position")
        # master_right = concat_state("observation.state.master_right_arm_position", "observation.state.master_right_gripper_position")
        # master_state = torch.cat([master_left, master_right], dim=-1).numpy()
        
        # The dataset's direct action could be used too, but the reference code forces action = master_state
        # action = master_state

        state_joint_position = data['observation.state.arm_joint_position']
        action_joint_position = data['action.arm_joint_position']
        # Assume action_joint_position has shape [Batch, 10]
        # Create a [Batch, 2] all-zero Tensor to concatenate at the end
        action_zeros_to_add = torch.zeros_like(action_joint_position[..., :2]) 
        # print("the action_zeros_to_add shape is:", action_zeros_to_add.shape)
        # print("the action_joint_position shape is:", action_joint_position.shape)
        # print("the state_joint_position shape is:", state_joint_position.shape)
        state_zeros_to_add = torch.zeros_like(state_joint_position[..., :2]) 
        action = torch.cat([action_joint_position, action_zeros_to_add], dim=-1).numpy()
        state = torch.cat([state_joint_position, state_zeros_to_add], dim=-1).numpy()

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


class OtherXHumanoidValueFunctionLerobotOfflineWrapperForBC(torch.utils.data.Dataset):
    """OtherXHumanoid value function wrapper without history frames"""
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

        # --- 1. Image preprocessing (Multi-Camera) ---
        # Use only the current frame, no history frames
        base_image = self._parse_image(data.get("observation.rgb_images.camera_head"))
        left_wrist_image = self._parse_image(data.get("observation.image.left", torch.zeros((3, 224, 224))))
        right_wrist_image = self._parse_image(data.get("observation.image.right", torch.zeros((3, 224, 224))))

        # --- 2. State and Action concatenation ---
        state_joint_position = data['observation.state.arm_joint_position']
        action_joint_position = data['action.arm_joint_position']
        # Assume action_joint_position has shape [Batch, 10]
        # Create a [Batch, 2] all-zero Tensor to concatenate at the end
        action_zeros_to_add = torch.zeros_like(action_joint_position[..., :2])
        state_zeros_to_add = torch.zeros_like(state_joint_position[..., :2])
        action = torch.cat([action_joint_position, action_zeros_to_add], dim=-1).numpy()
        state = torch.cat([state_joint_position, state_zeros_to_add], dim=-1).numpy()

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
                return_to_go_gt = data['return_to_go'].item() // 50
                return_to_go_gt = np.array(return_to_go_gt, dtype=np.int32)
                return_to_go_gt = np.clip(return_to_go_gt, 0, 10)
            else:
                return_to_go_gt = np.array(data['return_to_go'] // 50, dtype=np.int32)
                changed_return_to_go = None

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


class OtherOpenXValueFunctionLerobotOfflineWrapperForBC(torch.utils.data.Dataset):
    """OtherOpenX value function wrapper without history frames"""
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

        # --- 1. Image preprocessing (Multi-Camera) ---
        # Use only the current frame, no history frames
        base_image = self._parse_image(data.get("observation.images.image"))
        left_wrist_image = self._parse_image(data.get("observation.images.wrist_image", torch.zeros((3, 224, 224))))
        right_wrist_image = self._parse_image(data.get("observation.image.right", torch.zeros((3, 224, 224))))

        # --- 2. State and Action concatenation ---
        state_joint_position = data['observation.state']
        action_joint_position = data['action']
        # Assume action_joint_position has shape [Batch, 10]
        # Create a [Batch, 2] all-zero Tensor to concatenate at the end
        action_zeros_to_add = torch.zeros_like(action_joint_position[..., :2])
        state_zeros_to_add = torch.zeros_like(state_joint_position[..., :2])
        action = torch.cat([action_joint_position, action_zeros_to_add], dim=-1).numpy()
        state = torch.cat([state_joint_position, state_zeros_to_add], dim=-1).numpy()

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
                return_to_go_gt = data['return_to_go'].item() // 50
                return_to_go_gt = np.array(return_to_go_gt, dtype=np.int32)
                return_to_go_gt = np.clip(return_to_go_gt, 0, 10)
            else:
                return_to_go_gt = np.array(data['return_to_go'] // 50, dtype=np.int32)
                changed_return_to_go = None

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
                "left_wrist_0_rgb": np.True_ if "observation.images.wrist_image" in data.keys() else np.False_,
                "right_wrist_0_rgb": np.False_,
            },
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
