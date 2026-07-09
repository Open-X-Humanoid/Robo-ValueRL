import torch
import numpy as np
import random
class XHumanoidLerobotOfflineWrapperForBC(torch.utils.data.Dataset):
    def __init__(self, offline_dataset, sample_from_ratio=1, transforms=None, image_transform=None):
        self.offline_dataset = offline_dataset
        self.sample_from_ratio = sample_from_ratio
        origin_dataset_len = len(offline_dataset)
        self.sample_len = int(origin_dataset_len * sample_from_ratio)
        self.sample_indices = random.sample(range(origin_dataset_len), self.sample_len)
        self.transforms = transforms
        self.image_transform = image_transform
        
    
    def reset_sample_indices(self):
        self.sample_indices = random.sample(range(len(self.offline_dataset)), self.sample_len)

    def __len__(self):
        return self.sample_len

    def _parse_image(self, image_data):
        """Convert a (C, H, W) Tensor to a (H, W, C) Numpy array"""
        if isinstance(image_data, torch.Tensor):
            return image_data.permute(1, 2, 0).numpy()
        return image_data

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        data = self.offline_dataset[sample_idx]
        
        # --- 1. Image preprocessing (Multi-Camera) ---
        # Get images from three viewpoints, fill with a zero matrix if missing
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

        # Current state (Puppet)
        puppet_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        puppet_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        state = torch.cat([puppet_left, puppet_right], dim=-1).numpy()

        # Reference action (Master)
        master_left = concat_state("observation.state.master_left_arm_position", "observation.state.master_left_gripper_position")
        master_right = concat_state("observation.state.master_right_arm_position", "observation.state.master_right_gripper_position")
        action = torch.cat([master_left, master_right], dim=-1).numpy()

        if "quality" in data.keys():
            quality = data["quality"]
        else:
            quality = np.ones((action.shape[0], action.shape[1]), dtype=np.int64)


        # --- 4. Assemble structured data ---
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
            "action_quality": quality,
            "is_online_data": False,
        }

        # Include language instruction
        if "task" in data:
            align_data["prompt"] = data["task"]
        # print("the align_data is:", align_data.keys())

        # --- 5. Apply Transforms ---
        if self.transforms is not None:
            transformed_data = self.transforms(align_data)

            # Additional image augmentation (e.g. ColorJitter)
            if self.image_transform is not None:
                if 'image' in transformed_data:
                    for sub_key in transformed_data['image'].keys():
                        img = transformed_data['image'][sub_key]
                        # Keep dimensions consistent after augmentation
                        # print("the img is:", img.shape)
                        # img = img.transpose(1,2,0)
                        transformed_data['image'][sub_key] = self.image_transform(image=img)['image']
                        # transformed_data['image'][sub_key] = transformed_data['image'][sub_key].transpose(2,0,1)
            # print("the transformed_data is:", transformed_data)
            return transformed_data
        
        return align_data


class XHumanoidLerobotOfflineWrapperForBCWithHumanData(torch.utils.data.Dataset):
    def __init__(self, offline_dataset, sample_from_ratio=1, transforms=None, image_transform=None):
        self.offline_dataset = offline_dataset
        self.sample_from_ratio = sample_from_ratio
        origin_dataset_len = len(offline_dataset)
        self.sample_len = int(origin_dataset_len * sample_from_ratio)
        self.sample_indices = random.sample(range(origin_dataset_len), self.sample_len)
        self.transforms = transforms
        self.image_transform = image_transform
        
    
    def reset_sample_indices(self):
        self.sample_indices = random.sample(range(len(self.offline_dataset)), self.sample_len)

    def __len__(self):
        return self.sample_len

    def _parse_image(self, image_data):
        """Convert a (C, H, W) Tensor to a (H, W, C) Numpy array"""
        if isinstance(image_data, torch.Tensor):
            return image_data.permute(1, 2, 0).numpy()
        return image_data

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        data = self.offline_dataset[sample_idx]
        
        # --- 1. Image preprocessing (Multi-Camera) ---
        # Get images from three viewpoints, fill with a zero matrix if missing
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

        # Current state (Puppet)
        puppet_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        puppet_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        state = torch.cat([puppet_left, puppet_right], dim=-1).numpy()

        # Reference action (Master)
        master_left = concat_state("observation.state.master_left_arm_position", "observation.state.master_left_gripper_position")
        master_right = concat_state("observation.state.master_right_arm_position", "observation.state.master_right_gripper_position")
        action = torch.cat([master_left, master_right], dim=-1).numpy()

        if "quality" in data.keys():
            quality = data["quality"]
        else:
            quality = np.ones((action.shape[0], action.shape[1]), dtype=np.int64)

        if "human_intervention" in data.keys():
            human_intervention = data["human_intervention"]
            
            # here we inject human intervention in the future to indicate the quality of current action
            current_human_intervention = human_intervention[0]
            if current_human_intervention == 1:
                is_online_data = True
            else:
                is_online_data = False
        else:
            is_online_data = False
        # print("is_online_data is:",is_online_data)
        # print("the human intervention is:", data["human_intervention"])
        # --- 4. Assemble structured data ---
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
            "action_quality": quality,
            "is_online_data": is_online_data,
        }
        # print("the data item online is:", True)

        # Include language instruction
        if "task" in data:
            align_data["prompt"] = data["task"]
        # print("the align_data is:", align_data.keys())

        # --- 5. Apply Transforms ---
        if self.transforms is not None:
            transformed_data = self.transforms(align_data)

            # Additional image augmentation (e.g. ColorJitter)
            if self.image_transform is not None:
                if 'image' in transformed_data:
                    for sub_key in transformed_data['image'].keys():
                        img = transformed_data['image'][sub_key]
                        # Keep dimensions consistent after augmentation
                        # print("the img is:", img.shape)
                        # img = img.transpose(1,2,0)
                        transformed_data['image'][sub_key] = self.image_transform(image=img)['image']
                        # transformed_data['image'][sub_key] = transformed_data['image'][sub_key].transpose(2,0,1)
            # print("the transformed_data is:", transformed_data)
            return transformed_data
        
        return align_data

class XHumanoidLerobotOfflineWrapperForBCWithOnlineData(torch.utils.data.Dataset):
    def __init__(self, offline_dataset, sample_from_ratio=1, transforms=None, image_transform=None):
        self.offline_dataset = offline_dataset
        self.sample_from_ratio = sample_from_ratio
        origin_dataset_len = len(offline_dataset)
        self.sample_len = int(origin_dataset_len * sample_from_ratio)
        self.sample_indices = random.sample(range(origin_dataset_len), self.sample_len)
        self.transforms = transforms
        self.image_transform = image_transform
        
    
    def reset_sample_indices(self):
        self.sample_indices = random.sample(range(len(self.offline_dataset)), self.sample_len)

    def __len__(self):
        return self.sample_len

    def _parse_image(self, image_data):
        """Convert a (C, H, W) Tensor to a (H, W, C) Numpy array"""
        if isinstance(image_data, torch.Tensor):
            return image_data.permute(1, 2, 0).numpy()
        return image_data

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        data = self.offline_dataset[sample_idx]
        
        # --- 1. Image preprocessing (Multi-Camera) ---
        # Get images from three viewpoints, fill with a zero matrix if missing
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

        # Current state (Puppet)
        puppet_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        puppet_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        state = torch.cat([puppet_left, puppet_right], dim=-1).numpy()

        # Reference action (Master)
        master_left = concat_state("observation.state.master_left_arm_position", "observation.state.master_left_gripper_position")
        master_right = concat_state("observation.state.master_right_arm_position", "observation.state.master_right_gripper_position")
        action = torch.cat([master_left, master_right], dim=-1).numpy()

        if "quality" in data.keys():
            quality = data["quality"]
        else:
            quality = np.ones((action.shape[0], action.shape[1]), dtype=np.int64)

        # if "human_intervention" in data.keys():
        #     human_intervention = data["human_intervention"]
            
        #     # here we inject human intervention in the future to indicate the quality of current action
        #     current_human_intervention = human_intervention[0]
        #     if current_human_intervention == 1:
        #         is_online_data = True
        #     else:
        #         is_online_data = False
        # else:
        #     is_online_data = False
        # print("is_online_data is:",is_online_data)
        # print("the human intervention is:", data["human_intervention"])
        is_online_data = True
        # --- 4. Assemble structured data ---
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
            "action_quality": quality,
            "is_online_data": is_online_data,
        }
        # print("the data item online is:", True)

        # Include language instruction
        if "task" in data:
            align_data["prompt"] = data["task"]
        # print("the align_data is:", align_data.keys())

        # --- 5. Apply Transforms ---
        if self.transforms is not None:
            transformed_data = self.transforms(align_data)

            # Additional image augmentation (e.g. ColorJitter)
            if self.image_transform is not None:
                if 'image' in transformed_data:
                    for sub_key in transformed_data['image'].keys():
                        img = transformed_data['image'][sub_key]
                        # Keep dimensions consistent after augmentation
                        # print("the img is:", img.shape)
                        # img = img.transpose(1,2,0)
                        transformed_data['image'][sub_key] = self.image_transform(image=img)['image']
                        # transformed_data['image'][sub_key] = transformed_data['image'][sub_key].transpose(2,0,1)
            # print("the transformed_data is:", transformed_data)
            return transformed_data
        
        return align_data



class XHumanoidLerobotOfflineWrapperForBCWithOnlineValue(torch.utils.data.Dataset):
    def __init__(self, offline_dataset, sample_from_ratio=1, transforms=None, image_transform=None):
        self.offline_dataset = offline_dataset
        self.sample_from_ratio = sample_from_ratio
        origin_dataset_len = len(offline_dataset)
        self.sample_len = int(origin_dataset_len * sample_from_ratio)
        self.sample_indices = random.sample(range(origin_dataset_len), self.sample_len)
        self.transforms = transforms
        self.image_transform = image_transform
        
    
    def reset_sample_indices(self):
        self.sample_indices = random.sample(range(len(self.offline_dataset)), self.sample_len)

    def __len__(self):
        return self.sample_len

    def _parse_image(self, image_data):
        """Convert a (C, H, W) Tensor to a (H, W, C) Numpy array"""
        if isinstance(image_data, torch.Tensor):
            return image_data.permute(1, 2, 0).numpy()
        return image_data

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        data = self.offline_dataset[sample_idx]
        
        # --- 1. Image preprocessing (Multi-Camera) ---
        # Get images from three viewpoints, fill with a zero matrix if missing
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

        # Current state (Puppet)
        puppet_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        puppet_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        state = torch.cat([puppet_left, puppet_right], dim=-1).numpy()

        # Reference action (Master)
        master_left = concat_state("observation.state.master_left_arm_position", "observation.state.master_left_gripper_position")
        master_right = concat_state("observation.state.master_right_arm_position", "observation.state.master_right_gripper_position")
        action = torch.cat([master_left, master_right], dim=-1).numpy()

        if "quality" in data.keys():
            quality = data["quality"]
        else:
            quality = np.ones((action.shape[0], action.shape[1]), dtype=np.int64)

        # if "human_intervention" in data.keys():
        #     human_intervention = data["human_intervention"]
            
        #     # here we inject human intervention in the future to indicate the quality of current action
        #     current_human_intervention = human_intervention[0]
        #     if current_human_intervention == 1:
        #         is_online_data = True
        #     else:
        #         is_online_data = False
        # else:
        #     is_online_data = False
        # print("is_online_data is:",is_online_data)
        # print("the human intervention is:", data["human_intervention"])
        if quality[0] == 2:
            is_online_data = True
        else:
            is_online_data = False
        # --- 4. Assemble structured data ---
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
            "action_quality": quality,
            "is_online_data": is_online_data,
        }
        # print("the data item online is:", True)

        # Include language instruction
        if "task" in data:
            align_data["prompt"] = data["task"]
        # print("the align_data is:", align_data.keys())

        # --- 5. Apply Transforms ---
        if self.transforms is not None:
            transformed_data = self.transforms(align_data)

            # Additional image augmentation (e.g. ColorJitter)
            if self.image_transform is not None:
                if 'image' in transformed_data:
                    for sub_key in transformed_data['image'].keys():
                        img = transformed_data['image'][sub_key]
                        # Keep dimensions consistent after augmentation
                        # print("the img is:", img.shape)
                        # img = img.transpose(1,2,0)
                        transformed_data['image'][sub_key] = self.image_transform(image=img)['image']
                        # transformed_data['image'][sub_key] = transformed_data['image'][sub_key].transpose(2,0,1)
            # print("the transformed_data is:", transformed_data)
            return transformed_data
        
        return align_data



class XHumanoidLerobotOfflineWrapperForBCForElec(torch.utils.data.Dataset):
    def __init__(self, offline_dataset, sample_from_ratio=1, transforms=None, image_transform=None, target_subgoal_id=None):
        self.offline_dataset = offline_dataset
        self.sample_from_ratio = sample_from_ratio

        if target_subgoal_id is not None:
            all_task_indices = offline_dataset.hf_dataset['task_index']
            self._valid_indices = [i for i, t in enumerate(all_task_indices) if t == target_subgoal_id]
            if not self._valid_indices:
                unique_ids = sorted(set(int(x) for x in all_task_indices))
                raise ValueError(
                    f"target_subgoal_id={target_subgoal_id} matched no samples. "
                    f"Valid task_index values in this dataset: {unique_ids}"
                )
        else:
            self._valid_indices = list(range(len(offline_dataset)))

        self.sample_len = int(len(self._valid_indices) * sample_from_ratio)
        self.sample_indices = random.sample(self._valid_indices, self.sample_len)
        self.transforms = transforms
        self.image_transform = image_transform


    def reset_sample_indices(self):
        self.sample_indices = random.sample(self._valid_indices, self.sample_len)

    def __len__(self):
        return self.sample_len

    def _parse_image(self, image_data):
        """Convert a (C, H, W) Tensor to a (H, W, C) Numpy array"""
        if isinstance(image_data, torch.Tensor):
            return image_data.permute(1, 2, 0).numpy()
        return image_data

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        data = self.offline_dataset[sample_idx]
        
        # --- 1. Image preprocessing (Multi-Camera) ---
        # Get images from three viewpoints, fill with a zero matrix if missing
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

        # Current state (Puppet)
        puppet_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        puppet_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        state = torch.cat([puppet_left, puppet_right], dim=-1).numpy()

        # Reference action (Master)
        master_left = concat_state("observation.state.master_left_arm_position", "observation.state.master_left_gripper_position")
        master_right = concat_state("observation.state.master_right_arm_position", "observation.state.master_right_gripper_position")
        action = torch.cat([master_left, master_right], dim=-1).numpy()

        if "quality" in data.keys():
            quality = data["quality"]
        else:
            quality = np.ones((action.shape[0], action.shape[1]), dtype=np.int64)


        # --- 4. Assemble structured data ---
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
            "action_quality": quality,
        }

        # Include language instruction
        if "task" in data:
            align_data["prompt"] = data["task"]
        # print("the align_data is:", align_data.keys())

        # --- 5. Apply Transforms ---
        if self.transforms is not None:
            transformed_data = self.transforms(align_data)

            # Additional image augmentation (e.g. ColorJitter)
            if self.image_transform is not None:
                if 'image' in transformed_data:
                    for sub_key in transformed_data['image'].keys():
                        img = transformed_data['image'][sub_key]
                        # Keep dimensions consistent after augmentation
                        # print("the img is:", img.shape)
                        # img = img.transpose(1,2,0)
                        transformed_data['image'][sub_key] = self.image_transform(image=img)['image']
                        # transformed_data['image'][sub_key] = transformed_data['image'][sub_key].transpose(2,0,1)
            # print("the transformed_data is:", transformed_data)
            return transformed_data
        
        return align_data



class XHumanoidLerobotOfflineWrapperForBCForChip(torch.utils.data.Dataset):
    def __init__(self, offline_dataset, sample_from_ratio=1, transforms=None, image_transform=None):
        self.offline_dataset = offline_dataset
        self.sample_from_ratio = sample_from_ratio
        origin_dataset_len = len(offline_dataset)
        self.sample_len = int(origin_dataset_len * sample_from_ratio)
        self.sample_indices = random.sample(range(origin_dataset_len), self.sample_len)
        self.transforms = transforms
        self.image_transform = image_transform
        
    
    def reset_sample_indices(self):
        self.sample_indices = random.sample(range(len(self.offline_dataset)), self.sample_len)

    def __len__(self):
        return self.sample_len

    def _parse_image(self, image_data):
        """Convert a (C, H, W) Tensor to a (H, W, C) Numpy array"""
        if isinstance(image_data, torch.Tensor):
            return image_data.permute(1, 2, 0).numpy()
        return image_data

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        data = self.offline_dataset[sample_idx]
        
        # --- 1. Image preprocessing (Multi-Camera) ---
        # Get images from three viewpoints, fill with a zero matrix if missing
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

        # Current state (Puppet)
        puppet_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        puppet_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        state = torch.cat([puppet_left, puppet_right], dim=-1).numpy()

        state = state[0]

        # Reference action (Master)
        master_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        master_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        action = torch.cat([master_left, master_right], dim=-1).numpy()

        # print("the state is:", state.shape)
        # print("the action is:", action.shape)

        if "quality" in data.keys():
            quality = data["quality"]
        else:
            quality = np.ones((action.shape[0], action.shape[1]), dtype=np.int64)


        # --- 4. Assemble structured data ---
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
            "action_quality": quality,
        }

        # Include language instruction
        if "task" in data:
            align_data["prompt"] = data["task"]
        # print("the align_data is:", align_data.keys())

        # --- 5. Apply Transforms ---
        if self.transforms is not None:
            transformed_data = self.transforms(align_data)

            # Additional image augmentation (e.g. ColorJitter)
            if self.image_transform is not None:
                if 'image' in transformed_data:
                    for sub_key in transformed_data['image'].keys():
                        img = transformed_data['image'][sub_key]
                        # Keep dimensions consistent after augmentation
                        # print("the img is:", img.shape)
                        # img = img.transpose(1,2,0)
                        transformed_data['image'][sub_key] = self.image_transform(image=img)['image']
                        # transformed_data['image'][sub_key] = transformed_data['image'][sub_key].transpose(2,0,1)
            # print("the transformed_data is:", transformed_data)
            return transformed_data
        
        return align_data

class XHumanoidLerobotOfflineWrapperForBCWithSubgoal(torch.utils.data.Dataset):
    def __init__(self, offline_dataset, sample_from_ratio=1, transforms=None, image_transform=None):
        self.offline_dataset = offline_dataset
        self.sample_from_ratio = sample_from_ratio
        origin_dataset_len = len(offline_dataset)
        self.sample_len = int(origin_dataset_len * sample_from_ratio)
        self.sample_indices = random.sample(range(origin_dataset_len), self.sample_len)
        self.transforms = transforms
        self.image_transform = image_transform
        
    
    def reset_sample_indices(self):
        self.sample_indices = random.sample(range(len(self.offline_dataset)), self.sample_len)

    def __len__(self):
        return self.sample_len

    def _parse_image(self, image_data):
        """Convert a (C, H, W) Tensor to a (H, W, C) Numpy array"""
        if isinstance(image_data, torch.Tensor):
            return image_data.permute(1, 2, 0).numpy()
        return image_data

    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        sample_idx = self.sample_indices[idx]
        data = self.offline_dataset[sample_idx]
        
        # --- 1. Image preprocessing (Multi-Camera) ---
        # Get images from three viewpoints, fill with a zero matrix if missing
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

        # Current state (Puppet)
        puppet_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        puppet_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        state = torch.cat([puppet_left, puppet_right], dim=-1).numpy()

        # Reference action (Master)
        master_left = concat_state("observation.state.master_left_arm_position", "observation.state.master_left_gripper_position")
        master_right = concat_state("observation.state.master_right_arm_position", "observation.state.master_right_gripper_position")
        action = torch.cat([master_left, master_right], dim=-1).numpy()

        if "quality" in data.keys():
            quality = data["quality"]
        else:
            quality = np.ones((action.shape[0], action.shape[1]), dtype=np.int64)

        subgoal_index = 0


        if "subgoal_index" in data.keys():
            subgoal_index = data["subgoal_index"]
            subgoal_index = subgoal_index[-1]
        else:
            print("no subgoal index in the data")
        # --- 4. Assemble structured data ---
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
            "action_quality": quality,
            "subgoal_index": subgoal_index,
        }

        # Include language instruction
        if "task" in data:
            align_data["prompt"] = data["task"]
        
        # if "subgoal_text" in data:
        #     subgoal_text = data["subgoal_text"]
        #     align_data['prompt'] = align_data['prompt'] + "Subgoal: " + subgoal_text
        
        # print("the align_data prompt is:", align_data['prompt'])
        # print("the align_data is:", align_data.keys())

        # --- 5. Apply Transforms ---
        if self.transforms is not None:
            transformed_data = self.transforms(align_data)

            # Additional image augmentation (e.g. ColorJitter)
            if self.image_transform is not None:
                if 'image' in transformed_data:
                    for sub_key in transformed_data['image'].keys():
                        img = transformed_data['image'][sub_key]
                        # Keep dimensions consistent after augmentation
                        # print("the img is:", img.shape)
                        # img = img.transpose(1,2,0)
                        transformed_data['image'][sub_key] = self.image_transform(image=img)['image']
                        # transformed_data['image'][sub_key] = transformed_data['image'][sub_key].transpose(2,0,1)
            
            return transformed_data
        
        return align_data




class XHumanoidLerobotOfflineWrapperForBCWithIntervention(torch.utils.data.Dataset):
    def __init__(self, offline_dataset, sample_from_ratio=1, transforms=None, image_transform=None):
        self.offline_dataset = offline_dataset
        self.sample_from_ratio = sample_from_ratio
        origin_dataset_len = len(offline_dataset)
        self.sample_len = int(origin_dataset_len * sample_from_ratio)
        self.sample_indices = random.sample(range(origin_dataset_len), self.sample_len)
        self.transforms = transforms
        self.image_transform = image_transform
        
    
    def reset_sample_indices(self):
        self.sample_indices = random.sample(range(len(self.offline_dataset)), self.sample_len)

    def __len__(self):
        return self.sample_len

    def _parse_image(self, image_data):
        """Convert a (C, H, W) Tensor to a (H, W, C) Numpy array"""
        if isinstance(image_data, torch.Tensor):
            return image_data.permute(1, 2, 0).numpy()
        return image_data

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        data = self.offline_dataset[sample_idx]
        
        # --- 1. Image preprocessing (Multi-Camera) ---
        # Get images from three viewpoints, fill with a zero matrix if missing
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

        # Current state (Puppet)
        puppet_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        puppet_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        state = torch.cat([puppet_left, puppet_right], dim=-1).numpy()

        # Reference action (Master)
        master_left = concat_state("observation.state.master_left_arm_position", "observation.state.master_left_gripper_position")
        master_right = concat_state("observation.state.master_right_arm_position", "observation.state.master_right_gripper_position")
        action = torch.cat([master_left, master_right], dim=-1).numpy()

        # import pdb; pdb.set_trace()
        if "quality" in data.keys():
            quality = data["quality"]
        else:
            quality = np.ones((action.shape[0], action.shape[1]), dtype=np.int64)

        if "human_intervention" in data.keys():
            human_intervention = data["human_intervention"]
            
            # here we inject human intervention in the future to indicate the quality of current action
            current_human_intervention = human_intervention[0]
            near_future_human_intervention = human_intervention[1]
            far_future_human_intervention = human_intervention[2]
            # print("the current_human_intervention is:", current_human_intervention)
            if current_human_intervention == 1:
                # print("here i set the quality to 2")
                quality[0] = torch.tensor(2, dtype=torch.int64)
            elif near_future_human_intervention == 1:
                quality[0] = torch.tensor(0, dtype=torch.int64)
            elif far_future_human_intervention == 1:
                quality[0] = torch.tensor(1, dtype=torch.int64)
            else:
                quality[0] = quality[0]
            
            

        # --- 4. Assemble structured data ---
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
            "action_quality": quality,
            "is_online_data": True,
        }

        # Include language instruction
        if "task" in data:
            align_data["prompt"] = data["task"]
        # print("the align_data is:", align_data.keys())

        # --- 5. Apply Transforms ---
        if self.transforms is not None:
            transformed_data = self.transforms(align_data)

            # Additional image augmentation (e.g. ColorJitter)
            if self.image_transform is not None:
                if 'image' in transformed_data:
                    for sub_key in transformed_data['image'].keys():
                        img = transformed_data['image'][sub_key]
                        # Keep dimensions consistent after augmentation
                        # print("the img is:", img.shape)
                        # img = img.transpose(1,2,0)
                        transformed_data['image'][sub_key] = self.image_transform(image=img)['image']
                        # transformed_data['image'][sub_key] = transformed_data['image'][sub_key].transpose(2,0,1)
            
            return transformed_data
        
        return align_data



class XHumanoidLerobotOfflineWrapperForBCWithInterventionDagger(torch.utils.data.Dataset):
    def __init__(self, offline_dataset, sample_from_ratio=1, transforms=None, image_transform=None):
        self.offline_dataset = offline_dataset
        self.sample_from_ratio = sample_from_ratio
        origin_dataset_len = len(offline_dataset)
        self.sample_len = int(origin_dataset_len * sample_from_ratio)
        self.sample_indices = random.sample(range(origin_dataset_len), self.sample_len)
        self.transforms = transforms
        self.image_transform = image_transform
        
    
    def reset_sample_indices(self):
        self.sample_indices = random.sample(range(len(self.offline_dataset)), self.sample_len)

    def __len__(self):
        return self.sample_len

    def _parse_image(self, image_data):
        """Convert a (C, H, W) Tensor to a (H, W, C) Numpy array"""
        if isinstance(image_data, torch.Tensor):
            return image_data.permute(1, 2, 0).numpy()
        return image_data

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        data = self.offline_dataset[sample_idx]
        
        # --- 1. Image preprocessing (Multi-Camera) ---
        # Get images from three viewpoints, fill with a zero matrix if missing
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

        # Current state (Puppet)
        puppet_left = concat_state("observation.state.puppet_left_arm_position", "observation.state.puppet_left_gripper_position")
        puppet_right = concat_state("observation.state.puppet_right_arm_position", "observation.state.puppet_right_gripper_position")
        state = torch.cat([puppet_left, puppet_right], dim=-1).numpy()

        # Reference action (Master)
        master_left = concat_state("observation.state.master_left_arm_position", "observation.state.master_left_gripper_position")
        master_right = concat_state("observation.state.master_right_arm_position", "observation.state.master_right_gripper_position")
        action = torch.cat([master_left, master_right], dim=-1).numpy()

        # import pdb; pdb.set_trace()
        if "quality" in data.keys():
            quality = data["quality"]
        else:
            quality = np.ones((action.shape[0], action.shape[1]), dtype=np.int64)

        if "human_intervention" in data.keys():
            human_intervention = data["human_intervention"]
            
            # here we inject human intervention in the future to indicate the quality of current action
            current_human_intervention = human_intervention[0]
            near_future_human_intervention = human_intervention[1]
            far_future_human_intervention = human_intervention[2]
            # print("the current_human_intervention is:", current_human_intervention)
            if current_human_intervention == 1:
                # print("here i set the quality to 2")
                quality[0] = torch.tensor(2, dtype=torch.int64)
            elif near_future_human_intervention == 1:
                quality[0] = torch.tensor(0, dtype=torch.int64)
            elif far_future_human_intervention == 1:
                quality[0] = torch.tensor(1, dtype=torch.int64)
            else:
                quality[0] = torch.tensor(1, dtype=torch.int64)
            
            

        # --- 4. Assemble structured data ---
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
            "action_quality": quality,
        }

        # Include language instruction
        if "task" in data:
            align_data["prompt"] = data["task"]
        # print("the align_data is:", align_data.keys())

        # --- 5. Apply Transforms ---
        if self.transforms is not None:
            transformed_data = self.transforms(align_data)

            # Additional image augmentation (e.g. ColorJitter)
            if self.image_transform is not None:
                if 'image' in transformed_data:
                    for sub_key in transformed_data['image'].keys():
                        img = transformed_data['image'][sub_key]
                        # Keep dimensions consistent after augmentation
                        # print("the img is:", img.shape)
                        # img = img.transpose(1,2,0)
                        transformed_data['image'][sub_key] = self.image_transform(image=img)['image']
                        # transformed_data['image'][sub_key] = transformed_data['image'][sub_key].transpose(2,0,1)
            
            return transformed_data
        
        return align_data