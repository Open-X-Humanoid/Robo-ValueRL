import numpy as np
from lerobot.common.datasets.online_buffer import OnlineBuffer
UR_Online_DATA_SPEC = {
    'action.arm_joint_position': {'dtype':  np.dtype('float32'), 'shape': (7,)},
    'observation.state.arm_joint_position': {'dtype': np.dtype('float32'), 'shape': (7,)}, 
    'observation.rgb_images.camera_front': {'dtype': np.dtype('uint8'), 'shape': (480, 640, 3) },
    'observation.rgb_images.camera_wrist': {'dtype': np.dtype('uint8'), 'shape': (480, 640, 3) },
    'next_observation.state.arm_joint_position': {'dtype': np.dtype('float32'), 'shape': (7,)}, 
    'next_observation.rgb_images.camera_front': {'dtype': np.dtype('uint8'), 'shape': (480, 640, 3) },
    'next_observation.rgb_images.camera_wrist': {'dtype': np.dtype('uint8'), 'shape': (480, 640, 3) },   
    'reward': {'dtype': np.dtype('float32'), 'shape': (1,)},
    'reward_mask': {'dtype': np.dtype('bool'), 'shape': (1,)},
    'value_reward': {'dtype': np.dtype('float32'), 'shape': (1,)},
    'discounted_value_return': {'dtype': np.dtype('float32'), 'shape': (1,)},
}

X_HUMANOID_Online_DATA_SPEC = {
    # Observation state - puppet (execution arm)
    'observation.state.puppet_left_arm_position': {'dtype': np.dtype('float32'), 'shape': (7,)},
    'observation.state.puppet_right_arm_position': {'dtype': np.dtype('float32'), 'shape': (7,)},
    'observation.state.puppet_left_gripper_position': {'dtype': np.dtype('float32'), 'shape': (1,)},
    'observation.state.puppet_right_gripper_position': {'dtype': np.dtype('float32'), 'shape': (1,)},
    # Observation state - master (teleoperation arm)
    'observation.state.master_left_arm_position': {'dtype': np.dtype('float32'), 'shape': (7,)},
    'observation.state.master_right_arm_position': {'dtype': np.dtype('float32'), 'shape': (7,)},
    'observation.state.master_left_gripper_position': {'dtype': np.dtype('float32'), 'shape': (1,)},
    'observation.state.master_right_gripper_position': {'dtype': np.dtype('float32'), 'shape': (1,)},
    # Images - three cameras
    'observation.image.base_camera': {'dtype': np.dtype('uint8'), 'shape': (336, 336, 3)},
    'observation.image.left_wrist_camera': {'dtype': np.dtype('uint8'), 'shape': (336, 336, 3)},
    'observation.image.right_wrist_camera': {'dtype': np.dtype('uint8'), 'shape': (336, 336, 3)},
    # Actions
    'action.left_arm_position': {'dtype': np.dtype('float32'), 'shape': (7,)},
    'action.right_arm_position': {'dtype': np.dtype('float32'), 'shape': (7,)},
    'action.left_gripper_position': {'dtype': np.dtype('float32'), 'shape': (1,)},
    'action.right_gripper_position': {'dtype': np.dtype('float32'), 'shape': (1,)},
    # Reward-related
    'value_reward': {'dtype': np.dtype('float32'), 'shape': (1,)},
    'discounted_value_return': {'dtype': np.dtype('float32'), 'shape': (1,)},
    # Other
    'y_presses': {'dtype': np.dtype('int64'), 'shape': ()},
    # Human intervention label: 1 indicates this step is a human intervention, 0 indicates autonomous model control
    'intervention': {'dtype': np.dtype('int64'), 'shape': ()},
}

X_HUMANOID_DATA_SPEC = {
    'observation.state.puppet_left_arm_position': {'dtype': np.dtype('float32'), 'shape': (7,)},
    'observation.state.puppet_right_arm_position': {'dtype': np.dtype('float32'), 'shape': (7,)},
    'observation.state.puppet_left_gripper_position': {'dtype': np.dtype('float32'), 'shape': (1,)},
    'observation.state.puppet_right_gripper_position': {'dtype': np.dtype('float32'), 'shape': (1,)},
    'observation.state.master_left_gripper_position': {'dtype': np.dtype('float32'), 'shape': (1,)},
    'observation.state.master_right_gripper_position': {'dtype': np.dtype('float32'), 'shape': (1,)},
    'observation.state.master_left_arm_position': {'dtype': np.dtype('float32'), 'shape': (7,)},
    'observation.state.master_right_arm_position': {'dtype': np.dtype('float32'), 'shape': (7,)},
    'observation.image.image': {'dtype': np.dtype('uint8'), 'shape': (360, 640, 3) },
}

ONLINE_BUFFER_KEYS = [
    OnlineBuffer.INDEX_KEY,
    OnlineBuffer.FRAME_INDEX_KEY,
    OnlineBuffer.EPISODE_INDEX_KEY,
    OnlineBuffer.TIMESTAMP_KEY
]
