from features import METAWORLD_FEATURES
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import os
import numpy as np
import time
import pickle

def load_trajectory(trajectory_path, task_instruction):
    try:
        with open(trajectory_path, "rb") as f:
            data = pickle.load(f)
        trajectory = []
        images = data['images']
        actions = data['actions']
        observations = data['observations']
        
        for step in range(len(observations)):
            ee_pose = observations[step][:4].astype(np.float32)
            action_ee_pose = actions[step][:4].astype(np.float32)
            image = images[step].astype(np.uint8)

            step_data = {
                "observation.image.image": image,
                "observation.state.ee_pose": ee_pose,
                "action.ee_pose": action_ee_pose,
                "task": task_instruction,
            }
            trajectory.append(step_data)
        return trajectory
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        print(f"trajectory_path: {trajectory_path}")
        return None


if __name__ == "__main__":
    # task_name = input("Enter the task name: ")
    task_name = "window_open"
    # data_dir = f"dataset/{task_name}"
    data_dir = "/mnt/dataset/toago6/workspace/code/envs/Metaworld/scripts/demonstrations/window-open-v3"
    traj_list = []
    
    
    for file_name in os.listdir(data_dir):
        data_path = os.path.join(data_dir, file_name)
        traj_list.append(data_path)
    task_instruction = " ".join(task_name.split("_"))
    # output_path = f"lerobot_datasets/lerobot_dataset_{task_name}"
    output_path = f"/mnt/dataset/toago6/workspace/code/envs/Metaworld/lerobot_datasets/lerobot_dataset_{task_name}"
    dataset = LeRobotDataset.create(
        features=METAWORLD_FEATURES,
        root=output_path,
        repo_id = f"metaworld/lerobot_dataset_{task_name}",
        fps=30,
        robot_type="metaworld",
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
    