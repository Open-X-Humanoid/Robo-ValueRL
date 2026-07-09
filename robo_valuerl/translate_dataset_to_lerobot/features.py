REAL_WORLD_FEATURES = {
    "observation.image.image": {
        "dtype": "video",
        "shape": (360, 640, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.state.puppet_left_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    },
    "observation.state.puppet_right_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    },
    "observation.state.puppet_left_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["left_gripper"],
    },
    "observation.state.puppet_right_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["right_gripper"],
    },
    "observation.state.master_left_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["master_gripper"],
    },
    "observation.state.master_right_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["master_right_gripper"],
    },
    "observation.state.master_left_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["master_left_arm_0", "master_left_arm_1", "master_left_arm_2", "master_left_arm_3", "master_left_arm_4", "master_left_arm_5", "master_left_arm_6"],
    },
    "observation.state.master_right_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["master_right_arm_0", "master_right_arm_1", "master_right_arm_2", "master_right_arm_3", "master_right_arm_4", "master_right_arm_5", "master_right_arm_6"],
    },
    "action.left_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    },
    "action.right_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    },
    "action.left_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["left_gripper"],
    },
    "action.right_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["right_gripper"],
    },
}



UR5_REAL_WORLD_FEATURES = {
    "observation.image.front_image": {
        "dtype": "video",
        "shape": (336, 336, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.image.wrist_image": {
        "dtype": "video",
        "shape": (336, 336, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.image.left_image": {
        "dtype": "video",
        "shape": (336, 336, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.image.right_image": {
        "dtype": "video",
        "shape": (336, 336, 3),
        "names": ["height", "width", "rgb"],
    },
    
    "observation.state.puppet_arm_position": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
    },
    "observation.state.puppet_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["left_gripper"],
    },
    "observation.state.master_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["master_gripper"],
    },
    "observation.state.master_arm_position": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["master_left_arm_0", "master_left_arm_1", "master_left_arm_2", "master_left_arm_3", "master_left_arm_4", "master_left_arm_5"],
    },
    "action.arm_position": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
    },
    "action.gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["left_gripper"],
    },
    "trajectory_type": {
        "dtype": "int32",
        "shape": (1,),
        "names": ["trajectory_type"],
    },
}



X_HUMANOID_REAL_WORLD_FEATURES = {
    "observation.image.image": {
        "dtype": "video",
        "shape": (336, 336, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.image.left": {
        "dtype": "video",
        "shape": (336, 336, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.image.right": {
        "dtype": "video",
        "shape": (336, 336, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.state.puppet_left_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    },
    "observation.state.puppet_right_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    },
    "observation.state.puppet_left_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["left_gripper"],
    },
    "observation.state.puppet_right_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["right_gripper"],
    },
    "observation.state.master_left_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["master_gripper"],
    },
    "observation.state.master_right_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["master_right_gripper"],
    },
    "observation.state.master_left_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["master_left_arm_0", "master_left_arm_1", "master_left_arm_2", "master_left_arm_3", "master_left_arm_4", "master_left_arm_5", "master_left_arm_6"],
    },
    "observation.state.master_right_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["master_right_arm_0", "master_right_arm_1", "master_right_arm_2", "master_right_arm_3", "master_right_arm_4", "master_right_arm_5", "master_right_arm_6"],
    },
    "action.left_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    },
    "action.right_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    },
    "action.left_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["left_gripper"],
    },
    "action.right_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["right_gripper"],
    },
    "value_reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["value_reward"],
    },
    "discounted_value_return": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["discounted_value_return"],
    },
    "y_pressed": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["y_pressed"],
    },
    'quality': {
        "dtype": "bool",
        "shape": (1,),
        "names": ["quality"],
    },
}



NEW_X_HUMANOID_REAL_WORLD_FEATURES = {
    "observation.image.image": {
        "dtype": "video",
        "shape": (480, 480, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.image.left": {
        "dtype": "video",
        "shape": (480, 480, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.image.right": {
        "dtype": "video",
        "shape": (480, 480, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.state.puppet_left_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    },
    "observation.state.puppet_right_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    },
    "observation.state.puppet_left_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["left_gripper"],
    },
    "observation.state.puppet_right_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["right_gripper"],
    },
    "observation.state.master_left_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["master_gripper"],
    },
    "observation.state.master_right_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["master_right_gripper"],
    },
    "observation.state.master_left_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["master_left_arm_0", "master_left_arm_1", "master_left_arm_2", "master_left_arm_3", "master_left_arm_4", "master_left_arm_5", "master_left_arm_6"],
    },
    "observation.state.master_right_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["master_right_arm_0", "master_right_arm_1", "master_right_arm_2", "master_right_arm_3", "master_right_arm_4", "master_right_arm_5", "master_right_arm_6"],
    },
    "action.left_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    },
    "action.right_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    },
    "action.left_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["left_gripper"],
    },
    "action.right_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["right_gripper"],
    },
    "value_reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["value_reward"],
    },
    "discounted_value_return": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["discounted_value_return"],
    },
    "y_pressed": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["y_pressed"],
    },
}


METAWORLD_FEATURES = {
    "observation.image.image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.state.ee_pose": {
        "dtype": "float32",
        "shape": (4,),
        "names": ["x", "y", "z", "gripper"],
    },
    "action.ee_pose": {
        "dtype": "float32",
        "shape": (4,),
        "names": ["x", "y", "z", "gripper"],
    }
}