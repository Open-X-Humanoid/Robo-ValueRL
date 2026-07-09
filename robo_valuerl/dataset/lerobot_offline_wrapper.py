import torch
from config.data_config import UR_Online_DATA_SPEC
from config.data_config import ONLINE_BUFFER_KEYS
import random
class LerobotOfflineWrapper(torch.utils.data.Dataset):
    def __init__(self, offline_dataset, sample_from_ratio = 0.1, transforms = None):
        self.offline_dataset = offline_dataset
        self.sample_from_ratio = sample_from_ratio
        origin_dataset_len = len(offline_dataset)
        self.sample_len = int(origin_dataset_len * sample_from_ratio)
        self.sample_indices = random.sample(range(origin_dataset_len), self.sample_len)
        self.transforms = transforms

    def reset_sample_indices(self):
        self.sample_indices = random.sample(range(len(self.offline_dataset)), self.sample_len)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        data = self.offline_dataset[sample_idx]
        align_data = {}
        # print("the data is:", data.keys())
        for key in UR_Online_DATA_SPEC.keys():
            if "image" in key:
                align_data[key] = data[key].permute(1,2,0)
        
        gripper = data["observation.state.hand_joint_position"].unsqueeze(-1)
        state = torch.cat([data["observation.state.arm_joint_position"], gripper], dim=-1)
        align_data["observation.state.arm_joint_position"] = state
        
        next_gripper = data["next_observation.state.hand_joint_position"].unsqueeze(-1)
        next_state = torch.cat([data["next_observation.state.arm_joint_position"], next_gripper], dim=-1)
        align_data["next_observation.state.arm_joint_position"] = next_state
        
        action_gripper = data["action.hand_joint_position"].unsqueeze(-1)
        # print("the action.arm_joint_position is:", data["action.arm_joint_position"].shape)
        # print("the action.hand_joint_position is:", action_gripper.shape)
        action_state = torch.cat([data["action.arm_joint_position"], action_gripper], dim=-1)
        align_data["action.arm_joint_position"] = action_state
        # print("the action state is:", action_state.shape)
        
        reward = data["reward"]
        align_data["reward"] = reward
        
        for key in ONLINE_BUFFER_KEYS:
            align_data[key] = data[key]
        
        if self.transforms is not None:
            return self.transforms(align_data)
        else:   
            return align_data