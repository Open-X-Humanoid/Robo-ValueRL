import argparse
import os
import cv2
import numpy as np
import time
import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from camera_utils import CameraUtils
from features import NEW_X_HUMANOID_REAL_WORLD_FEATURES

def load_trajectory(trajectory_path, task_instruction):
    try:
        data = h5py.File(trajectory_path, "r")
        trajectory = []
        simple_data = data['master']['arm_left_position_align']['data']
        timesteps = len(simple_data)
        discount = 1
        value_reward = np.ones((timesteps, 1)) * (-0.1)
        y_pressed = data['y_pressed'][:]
        y_pressed = y_pressed * 50
        
        # Cumulative form
        y_pressed_cumsum = np.cumsum(y_pressed)
        
        if "fail" in trajectory_path:
            value_reward[-1:] = -600
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
            # Get action data
            action_left_arm_position = data['master']['arm_left_position_align']['data'][step] 
            action_right_arm_position = data['master']['arm_right_position_align']['data'][step]
            action_left_gripper_position = data['master']['end_effector_left_position_align']['data'][step]
            action_right_gripper_position = data['master']['end_effector_right_position_align']['data'][step]
            
            # Image processing
            head_image = CameraUtils.decode_color_image(data['camera_observations']['color_images']['camera_top'][step], input_format="bgr", output_format="rgb")
            left_image = CameraUtils.decode_color_image(data['camera_observations']['color_images']['camera_left'][step], input_format="bgr", output_format="rgb")
            right_image = CameraUtils.decode_color_image(data['camera_observations']['color_images']['camera_right'][step], input_format="bgr", output_format="rgb")
            
            head_image = cv2.resize(head_image, (336, 336))
            left_image = cv2.resize(left_image, (336, 336))
            right_image = cv2.resize(right_image, (336, 336))
            
            master_left_gripper_position = data['master']['end_effector_left_position_align']['data'][step]
            master_right_gripper_position = data['master']['end_effector_right_position_align']['data'][step]

            # Puppet state handling
            if "end_effector_left_position_align" not in data['puppet'].keys():
                puppet_left_gripper_position = master_left_gripper_position
            else:
                puppet_left_gripper_position = data['puppet']['end_effector_left_position_align']['data'][step]
            
            if "end_effector_right_position_align" not in data['puppet'].keys():
                puppet_right_gripper_position = master_right_gripper_position
            else:
                puppet_right_gripper_position = data['puppet']['end_effector_right_position_align']['data'][step]
            
            step_data = {
                "observation.image.image": head_image,
                "observation.image.left": left_image,
                "observation.image.right": right_image,
                "observation.state.puppet_left_arm_position": np.array(data['puppet']['arm_left_position_align']['data'][step], dtype=np.float32),
                "observation.state.puppet_right_arm_position": np.array(data['puppet']['arm_right_position_align']['data'][step], dtype=np.float32),
                "observation.state.puppet_left_gripper_position": np.array(puppet_left_gripper_position, dtype=np.float32),
                "observation.state.puppet_right_gripper_position": np.array(puppet_right_gripper_position, dtype=np.float32),
                "observation.state.master_left_gripper_position": np.array(master_left_gripper_position, dtype=np.float32),     
                "observation.state.master_right_gripper_position": np.array(master_right_gripper_position, dtype=np.float32),
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

def main():
    parser = argparse.ArgumentParser(description="Convert X-Humanoid HDF5 data to LeRobot format.")
    
    # Add command-line arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Input directory containing .hdf5 trajectories")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the LeRobot dataset")
    parser.add_argument("--task_name", type=str, default="separate_the_blocks_and_sort_by_color_into_different_plates", 
                        help="Name of the task (used for repo_id and instructions)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second of the dataset")

    args = parser.parse_args()

    # Process task instruction
    task_instruction = " ".join(args.task_name.split("_"))
    
    # Initialize LeRobot dataset
    dataset = LeRobotDataset.create(
        features=NEW_X_HUMANOID_REAL_WORLD_FEATURES,
        root=args.output_path,
        repo_id=f"x_humanoid/lerobot_dataset_{args.task_name}",
        fps=args.fps,
        robot_type="x_humanoid",
    )

    # Recursively find all .hdf5 files
    hdf5_files = []
    for root, dirs, files in os.walk(args.data_dir):
        for file in files:
            if file.endswith(".hdf5"):
                hdf5_files.append(os.path.join(root, file))
    
    print(f"Found {len(hdf5_files)} HDF5 files in {args.data_dir}")

    for traj_index, traj_path in enumerate(hdf5_files):
        print(f"Processing trajectory {traj_index + 1}/{len(hdf5_files)}: {traj_path}")
        
        start_time = time.time()
        trajectory = load_trajectory(traj_path, task_instruction)
        
        if trajectory is None:
            continue
            
        print(f"Load time: {time.time() - start_time:.2f}s")
        
        for single_step in trajectory:
            dataset.add_frame(single_step)
        
        dataset.save_episode()
        print(f"Done with trajectory {traj_index}")

if __name__ == "__main__":
    main()