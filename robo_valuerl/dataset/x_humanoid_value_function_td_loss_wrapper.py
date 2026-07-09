import torch
import numpy as np
from openpi.models.hl_gauss_distribution import HLGaussDistribution
import torch
import numpy as np
class XHumanoidValueFunctionLerobotOfflineWrapperForTDLoss(torch.utils.data.Dataset):
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

    def _parse_image_at(self, image_data, t=0):
        """Take frame t from (T,C,H,W) or (C,H,W) and convert it to (H,W,C) numpy"""
        if isinstance(image_data, torch.Tensor):
            if image_data.dim() == 4:
                # (T, C, H, W) -> take timestep t -> (H, W, C)
                return image_data[t].permute(1, 2, 0).numpy()
            elif image_data.dim() == 3:
                # (C, H, W) -> (H, W, C)
                return image_data.permute(1, 2, 0).numpy()
        return image_data

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        data = self.offline_dataset[sample_idx]

        # --- 1. Image preprocessing (Multi-Camera) ---
        # delta_timestamps contains [0, 10/30] two frames, shape is (2, C, H, W)
        base_image_raw = data.get("observation.image.image", torch.zeros((3, 224, 224)))
        left_wrist_image_raw = data.get("observation.image.left", torch.zeros((3, 224, 224)))
        right_wrist_image_raw = data.get("observation.image.right", torch.zeros((3, 224, 224)))

        # Current frame (t=0)
        base_image = self._parse_image_at(base_image_raw, 0)
        left_wrist_image = self._parse_image_at(left_wrist_image_raw, 0)
        right_wrist_image = self._parse_image_at(right_wrist_image_raw, 0)

        # Next frame (t=1), only valid when the data contains multiple frames
        has_next_image = (isinstance(base_image_raw, torch.Tensor) and base_image_raw.dim() == 4
                          and base_image_raw.shape[0] >= 2)
        if has_next_image:
            next_base_image = self._parse_image_at(base_image_raw, 1)
            next_left_wrist_image = self._parse_image_at(left_wrist_image_raw, 1)
            next_right_wrist_image = self._parse_image_at(right_wrist_image_raw, 1)

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

        # Is Trajectory End
        is_traj_end = None
        if 'remain_time' in data:
            is_traj_end = np.array(int(data['remain_time']) in (0, 1, 2), dtype=np.bool_)

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
            "is_traj_end": is_traj_end,
        }

        # Add next_image / next_image_mask (for TD-loss)
        if has_next_image:
            align_data["next_image"] = {
                "base_0_rgb": next_base_image,
                "left_wrist_0_rgb": next_left_wrist_image,
                "right_wrist_0_rgb": next_right_wrist_image,
            }
            align_data["next_image_mask"] = {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
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
                for key in ['image', 'next_image']:
                    if key in transformed_data:
                        for sub_key in transformed_data[key].keys():
                            img = transformed_data[key][sub_key]
                            # Note: A.Compose usually expects uint8 HWC format
                            transformed_data[key][sub_key] = self.image_transform(image=img)['image']
            return transformed_data
        
        return align_data

