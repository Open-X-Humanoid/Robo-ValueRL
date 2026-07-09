import argparse
import os
import cv2
import numpy as np
import time
import h5py
from features import X_HUMANOID_REAL_WORLD_FEATURES
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from camera_utils import CameraUtils

def load_trajectory(trajectory_path, task_instruction):
    try:
        data = h5py.File(trajectory_path, "r")
        trajectory = []
        simple_data = data['master']['arm_left_position_align']['data']
        timesteps = len(simple_data)
        discount = 1
        value_reward = np.ones((timesteps, 1)) * (-0.1)
        
        # Process y_pressed reward
        y_pressed = data['y_pressed'][:]
        y_pressed = y_pressed * 50
        y_pressed_cumsum = np.cumsum(y_pressed)
        
        # Determine success/failure/suboptimal from the file name
        if "fail" in trajectory_path:
            value_reward[-1:] = -600
        else:
            value_reward[-1:] = 100
            
        quality = "suboptimal" not in trajectory_path
        quality_arr = np.array([quality], dtype=bool)
        
        # Compute discounted return
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
            # Image processing and resize
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
                "action.left_arm_position": np.array(data['master']['arm_left_position_align']['data'][step], dtype=np.float32),
                "action.right_arm_position": np.array(data['master']['arm_right_position_align']['data'][step], dtype=np.float32),
                "action.left_gripper_position": np.array(data['master']['end_effector_left_position_align']['data'][step], dtype=np.float32),
                "action.right_gripper_position": np.array(data['master']['end_effector_right_position_align']['data'][step], dtype=np.float32),
                "task": task_instruction,
                "value_reward": value_reward[step],
                "discounted_value_return": discounted_return[step],
                "y_pressed": y_pressed[step:step+1],
                "quality": quality_arr
            }
            trajectory.append(step_data)
        return trajectory
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 robot trajectories to LeRobot format.")
    
    # Command-line argument definitions
    parser.add_argument("--task_name", type=str, default="separate_the_blocks_and_sort_by_color_into_different_plates", help="Name of the task")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Directory containing the raw HDF5 trajectories")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the LeRobot dataset")
    parser.add_argument("--repo_id", type=str, default=None, help="Repo ID for the dataset (optional)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")

    args = parser.parse_args()

    # Path handling
    task_instruction = " ".join(args.task_name.split("_"))
    repo_id = args.repo_id if args.repo_id else f"x_humanoid/lerobot_dataset_{args.task_name}"
    
    # Find trajectory files
    traj_list = []
    if os.path.exists(args.raw_data_dir):
        for type_name in os.listdir(args.raw_data_dir):
            type_path = os.path.join(args.raw_data_dir, type_name)
            if os.path.isdir(type_path):
                for file_name in os.listdir(type_path):
                    data_path = os.path.join(type_path, file_name, "trajectory.hdf5")
                    if os.path.exists(data_path):
                        traj_list.append(data_path)
    
    if not traj_list:
        print(f"No trajectories found in {args.raw_data_dir}")
        return

    print(f"Found {len(traj_list)} trajectories. Starting conversion...")

    # Initialize LeRobot dataset
    dataset = LeRobotDataset.create(
        features=X_HUMANOID_REAL_WORLD_FEATURES,
        root=args.output_dir,
        repo_id=repo_id,
        fps=args.fps,
        robot_type="x_humanoid",
    )

    for traj_index, traj_path in enumerate(traj_list):
        start_time = time.time()
        trajectory = load_trajectory(traj_path, task_instruction)
        
        if trajectory is None:
            continue
            
        for single_step in trajectory:
            dataset.add_frame(single_step)
        
        dataset.save_episode()
        print(f"Done with trajectory {traj_index} | Time: {time.time() - start_time:.2f}s | Path: {traj_path}")

    print(f"Dataset conversion complete. Saved to: {args.output_dir}")

if __name__ == "__main__":
    main()