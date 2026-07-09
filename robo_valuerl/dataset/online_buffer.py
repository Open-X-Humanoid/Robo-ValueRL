import numpy as np
from config.data_config import ONLINE_BUFFER_KEYS
from lerobot.common.datasets.online_buffer import OnlineBuffer
import cv2

def create_online_buffer(temp_dir, data_spec, buffer_capacity, delta_timestamps, fps):
    return OnlineBuffer(
        write_dir=temp_dir,
        fps = fps,
        data_spec=data_spec,
        buffer_capacity=buffer_capacity,
        delta_timestamps=delta_timestamps,
    )
    
class RealWorldOnlineBuffer:
    def __init__(self, temp_dir, data_spec, buffer_capacity, fps = 30, transforms = None, delta_timestamps = None, image_transform = None):
        self.online_buffer = create_online_buffer(temp_dir, data_spec, buffer_capacity, delta_timestamps, fps)
        self.traj_num = self.online_buffer.num_episodes
        self.frame_num = len(self.online_buffer)
        self.fps = fps
        self.frequency = 1 / fps
        self.data_spec = data_spec
        
        self.timestamp = 0
        self.fram_index = 0
        self._single_traj_buffer = {
            key: [] for key in data_spec.keys()
        }
        for key in ONLINE_BUFFER_KEYS:
            self._single_traj_buffer[key] = []
        self.transforms = transforms
        self.image_transform = image_transform
        
    def __len__(self):
        return len(self.online_buffer)
    
    def __getitem__(self, idx):
        
        data = self.online_buffer[idx]
        trans_data = {
            key: data[key] for key in self.data_spec.keys()
        }
        trans_data.update({
            key: data[key] for key in ONLINE_BUFFER_KEYS
        })

        if self.transforms is not None:
            data =  self.transforms(trans_data)
            if self.image_transform is not None:
                for key in data.keys():
                    if 'image' in key and 'mask' not in key:
                        for sub_key in data[key].keys():
                            image = data[key][sub_key].transpose(1,2,0)
                            transformed_image = self.image_transform(image = image)['image']
                            # print("the transformed_image is:", transformed_image)
                            data[key][sub_key] = transformed_image.transpose(2,0,1)
            return data
        else:   
            return trans_data
    
    def _save_step(self, current_obs, current_action, next_obs, is_intervention=False):
        merged_obs = self.merge_step_into_observation(current_obs, current_action, next_obs, is_intervention=is_intervention)
        self.frame_num += 1
        self.fram_index += 1
        self.timestamp += self.frequency
        
        for key in merged_obs.keys():
            self._single_traj_buffer[key].append(merged_obs[key])
    
        
    
    def mark_trajectory_success(self, success=True):
        """Mark the last frame of the trajectory as success (1) or failure (0)"""
        if len(self._single_traj_buffer['y_presses']) > 0:
            self._single_traj_buffer['y_presses'][-1] = np.array(1 if success else 0, dtype=np.int64)
    
    def save_trajectory(self, success=None):
        """
        Save the trajectory
        
        Args:
            success: if specified, mark the trajectory as success (True) or failure (False)
        """
        if success is not None:
            self.mark_trajectory_success(success)
        # import pdb; pdb.set_trace()
        self.online_buffer.add_data(self._single_traj_buffer)
        self._single_traj_buffer = {
            key: [] for key in self.data_spec.keys()
        }
        for key in ONLINE_BUFFER_KEYS:
            self._single_traj_buffer[key] = []
        
        self.fram_index = 0
        self.traj_num += 1
        self.timestamp = 0
        print("the traj num is:", self.traj_num)

    def clear_single_traj_buffer(self):
        self._single_traj_buffer = {
            key: [] for key in self.data_spec.keys()
        }
        for key in ONLINE_BUFFER_KEYS:
            self._single_traj_buffer[key] = []
        self.fram_index = 0

    def merge_step_into_observation(self, current_obs, current_action, next_obs, is_intervention=False):
        """
        Merge observation, action, and the next observation
        
        State layout (16 dims): 7+1+7+1
        - left_arm:  0:7      (7 dims)
        - left_gripper: 7      (1 dim)
        - right_arm:  8:15     (7 dims)
        - right_gripper: 15    (1 dim)
        """
        # Extract the current and next state (dual-arm robot)
        cur_state = current_obs['state'][0]
        next_state = next_obs['state'][0]
        
        # Split left/right arm and gripper positions (7+1+7+1 order)
        cur_puppet_left_arm = cur_state[0:7]             # 0~6
        cur_puppet_left_gripper = cur_state[7:8]         # 7
        cur_puppet_right_arm = cur_state[8:15]           # 8~14
        cur_puppet_right_gripper = cur_state[15:16]      # 15
        
        # master state (filled from action, same sampling range as above)
        cur_master_left_arm = current_action[0:7]
        cur_master_left_gripper = current_action[7:8]
        cur_master_right_arm = current_action[8:15]
        cur_master_right_gripper = current_action[15:16]
        
        # Split out the action
        action_left_arm = current_action[0:7]
        action_left_gripper = current_action[7:8]
        action_right_arm = current_action[8:15]
        action_right_gripper = current_action[15:16]
        
        # Extract and resize the three camera images to 336x336
        base_camera = current_obs['image']['base_0_rgb'][0]
        left_wrist_camera = current_obs['image']['left_wrist_0_rgb'][0]
        right_wrist_camera = current_obs['image']['right_wrist_0_rgb'][0]
        
        base_camera = cv2.resize(base_camera, (336, 336)).astype(np.uint8)
        left_wrist_camera = cv2.resize(left_wrist_camera, (336, 336)).astype(np.uint8)
        right_wrist_camera = cv2.resize(right_wrist_camera, (336, 336)).astype(np.uint8)
        
        merged_obs = {
            # Observation state - puppet (execution arm)
            'observation.state.puppet_left_arm_position': cur_puppet_left_arm.astype(np.float32),
            'observation.state.puppet_left_gripper_position': cur_puppet_left_gripper.astype(np.float32),
            'observation.state.puppet_right_arm_position': cur_puppet_right_arm.astype(np.float32),
            'observation.state.puppet_right_gripper_position': cur_puppet_right_gripper.astype(np.float32),
            # Observation state - master (teleoperation arm)
            'observation.state.master_left_arm_position': cur_master_left_arm.astype(np.float32),
            'observation.state.master_left_gripper_position': cur_master_left_gripper.astype(np.float32),
            'observation.state.master_right_arm_position': cur_master_right_arm.astype(np.float32),
            'observation.state.master_right_gripper_position': cur_master_right_gripper.astype(np.float32),
            # Image - three cameras
            'observation.image.base_camera': base_camera,
            'observation.image.left_wrist_camera': left_wrist_camera,
            'observation.image.right_wrist_camera': right_wrist_camera,
            # Action
            'action.left_arm_position': action_left_arm.astype(np.float32),
            'action.left_gripper_position': action_left_gripper.astype(np.float32),
            'action.right_arm_position': action_right_arm.astype(np.float32),
            'action.right_gripper_position': action_right_gripper.astype(np.float32),
            # Reward related
            "value_reward": np.array([0.0], dtype=np.float32),
            "discounted_value_return": np.array([0.0], dtype=np.float32),
            # Other
            "y_presses": np.array(0, dtype=np.int64),
            # Human intervention label: 1 means this step is human intervention, 0 means automatic model control
            "intervention": np.array(1 if is_intervention else 0, dtype=np.int64),
            OnlineBuffer.INDEX_KEY: np.array(self.fram_index, dtype=np.int64),
            OnlineBuffer.FRAME_INDEX_KEY: np.array(self.fram_index, dtype=np.int64),
            OnlineBuffer.EPISODE_INDEX_KEY: np.array(0, dtype=np.int64),
            OnlineBuffer.TIMESTAMP_KEY: np.array(self.timestamp, dtype=np.float64)
        }
        return merged_obs

