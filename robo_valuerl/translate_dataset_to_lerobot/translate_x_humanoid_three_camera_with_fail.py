from features import X_HUMANOID_REAL_WORLD_FEATURES
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import h5py
from camera_utils import CameraUtils
import os
import cv2
import numpy as np
import time
def load_trajectory(trajectory_path, task_instruction):
    try:
        data = h5py.File(trajectory_path, "r")
        trajectory = []
        simple_data = data['master']['arm_left_position_align']['data']
        timesteps = len(simple_data)
        discount = 1
        value_reward = np.ones((timesteps,1 )) * (-0.1)
        y_pressed = data['y_pressed'][:]
        y_pressed = y_pressed * 50
        
        # Cumulative form
        y_pressed_cumsum = np.cumsum(y_pressed)
        print("y_pressed_cumsum shape is", y_pressed_cumsum.shape)
        if "fail" in trajectory_path:
            value_reward[-1:] = -300
        else:
            value_reward[-1:] = 100
        
        discounted_return = np.zeros_like(value_reward)
        discounted_return[-1] = value_reward[-1]
        for i in range(timesteps-2, -1, -1):
            discounted_return[i] = value_reward[i] + discount * discounted_return[i+1]
        
        for i in range(timesteps):
            discounted_return[i] = discounted_return[i] + y_pressed_cumsum[i]

        
        value_reward = value_reward.astype(np.float32)
        discounted_return = discounted_return.astype(np.float32)
        y_pressed = y_pressed.astype(np.float32)
        
        for step in range(timesteps):
            action_left_arm_position = data['master']['arm_left_position_align']['data'][step] 
            action_right_arm_position = data['master']['arm_right_position_align']['data'][step]
            action_left_gripper_position = data['master']['end_effector_left_position_align']['data'][step]
            action_right_gripper_position = data['master']['end_effector_right_position_align']['data'][step]
            # import pdb; pdb.set_trace()
            head_image = CameraUtils.decode_color_image(data['camera_observations']['color_images']['camera_top'][step], input_format="bgr", output_format="rgb")
            left_image = CameraUtils.decode_color_image(data['camera_observations']['color_images']['camera_left'][step], input_format="bgr", output_format="rgb")
            right_image = CameraUtils.decode_color_image(data['camera_observations']['color_images']['camera_right'][step], input_format="bgr", output_format="rgb")
            # print("previous head image shape is", head_image.shape)
            head_image = cv2.resize(head_image, (336, 336))
            left_image = cv2.resize(left_image, (336, 336))
            right_image = cv2.resize(right_image, (336, 336))

            step_data = {
                "observation.image.image": head_image,
                "observation.image.left": left_image,
                "observation.image.right": right_image,
                "observation.state.puppet_left_arm_position": np.array(data['puppet']['arm_left_position_align']['data'][step], dtype=np.float32),
                "observation.state.puppet_right_arm_position": np.array(data['puppet']['arm_right_position_align']['data'][step], dtype=np.float32),
                "observation.state.puppet_left_gripper_position": np.array(data['puppet']['end_effector_left_position_align']['data'][step], dtype=np.float32),
                "observation.state.puppet_right_gripper_position": np.array(data['puppet']['end_effector_right_position_align']['data'][step], dtype=np.float32),
                "observation.state.master_left_gripper_position": np.array(data['master']['end_effector_left_position_align']['data'][step], dtype=np.float32),     
                "observation.state.master_right_gripper_position": np.array(data['master']['end_effector_right_position_align']['data'][step], dtype=np.float32),
                "observation.state.master_left_arm_position": np.array(data['master']['arm_left_position_align']['data'][step], dtype=np.float32),
                "observation.state.master_right_arm_position": np.array(data['master']['arm_right_position_align']['data'][step], dtype=np.float32),
                "action.left_arm_position": np.array(action_left_arm_position, dtype=np.float32),
                "action.right_arm_position": np.array(action_right_arm_position, dtype=np.float32),
                "action.left_gripper_position": np.array(action_left_gripper_position, dtype=np.float32),
                "action.right_gripper_position": np.array(action_right_gripper_position, dtype=np.float32),
                "task": task_instruction,
                "value_reward": value_reward[step],
                "discounted_value_return": discounted_return[step],
                "y_pressed": y_pressed[step:step+1]
            }
            trajectory.append(step_data)
        return trajectory
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        print(f"trajectory_path: {trajectory_path}")
        return None

if __name__ == "__main__":
    # task_name = input("Enter the task name: ")
    task_name = "separate_the_blocks_and_sort_by_color_into_different_plates"
    # data_dir = f"dataset/{task_name}"
    data_dir = "/mnt/dataset/toago6/block_x_humanoid_data/origin/16_24_data"
    traj_list = []
    
    
    
    for type_name in os.listdir(data_dir):
        for file_name in os.listdir(os.path.join(data_dir, type_name)):
            data_path = os.path.join(data_dir, type_name, file_name, "trajectory.hdf5")
            traj_list.append(data_path)
    task_instruction = " ".join(task_name.split("_"))
    # output_path = f"lerobot_datasets/lerobot_dataset_{task_name}"
    output_path = "/home/wink/x_humanoid_lerobot_data_without_discount_24"
    dataset = LeRobotDataset.create(
        features=X_HUMANOID_REAL_WORLD_FEATURES,
        root=output_path,
        repo_id = f"x_humanoid/lerobot_dataset_{task_name}",
        fps=30,
        robot_type="x_humanoid",
    )
    # dataset.save_dataset()
    for traj_index, traj_path in enumerate(traj_list):
        start_time = time.time()
        trajectory = load_trajectory(traj_path, task_instruction)
        if trajectory is None:
            continue
        print(f"load_trajectory_time: {time.time() - start_time}")
        for single_step in trajectory:
            add_frame_time = time.time()
            dataset.add_frame(
                single_step,
                # task = task_instruction
            )
            end_add_frame_time = time.time()
            # print(f"add_frame_time: {end_add_frame_time - add_frame_time}")
        
        print("done with trajectory", traj_index)
        dataset.save_episode()
    