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
    "observation.state.master_left_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    },
    "observation.state.master_right_arm_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    },
    "observation.state.master_left_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["left_gripper"],
    },
    "observation.state.master_right_gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["right_gripper"],
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
    "reward":{
        "dtype": "float32",
        "shape": (1,),
        "names": ["reward"],
    },
    "reward_mask":{
        "dtype": "bool",
        "shape": (1,),
        "names": ["reward_mask"],
    },
    "value_reward":{
        "dtype": "float32",
        "shape": (1,),
        "names": ["value_reward"],
    },
    "discounted_value_return":{
        "dtype": "float32",
        "shape": (1,),
        "names": ["discounted_value_return"],
    },
    "human_intervention": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["human_intervention"],
    },
}
