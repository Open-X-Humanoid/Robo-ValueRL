from dataset.online_buffer import RealWorldOnlineBuffer
from config.lerobot_features import X_HUMANOID_REAL_WORLD_FEATURES
from config.data_config import X_HUMANOID_Online_DATA_SPEC
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def load_trajectory(traj_n, online_buffer, st_idx):
    """
    Load one trajectory from online_buffer

    The new online_buffer format directly stores separated states and actions:
    - observation.state.puppet_left_arm_position (7-dim)
    - observation.state.puppet_right_arm_position (7-dim)
    - observation.state.puppet_left_gripper_position (1-dim)
    - observation.state.puppet_right_gripper_position (1-dim)
    - observation.state.master_left_arm_position (7-dim)
    - observation.state.master_right_arm_position (7-dim)
    - observation.state.master_left_gripper_position (1-dim)
    - observation.state.master_right_gripper_position (1-dim)
    - action.left_arm_position (7-dim)
    - action.right_arm_position (7-dim)
    - action.left_gripper_position (1-dim)
    - action.right_gripper_position (1-dim)
    """
    trajectory = []
    traj_len = 0
    temp_idx = st_idx
    
    while True:
        if temp_idx >= len(online_buffer):
            print("reach the end of the online buffer")
            break
        episode_index = online_buffer[temp_idx]['episode_index']
        if episode_index != traj_n:
            break
        
        # Directly read the already-separated states and actions
        observation_state_puppet_left_arm_position = online_buffer[temp_idx]['observation.state.puppet_left_arm_position'].numpy()
        observation_state_puppet_left_gripper_position = online_buffer[temp_idx]['observation.state.puppet_left_gripper_position'].numpy()
        observation_state_puppet_right_arm_position = online_buffer[temp_idx]['observation.state.puppet_right_arm_position'].numpy()
        observation_state_puppet_right_gripper_position = online_buffer[temp_idx]['observation.state.puppet_right_gripper_position'].numpy()
        
        # Read master state
        observation_state_master_left_arm_position = online_buffer[temp_idx]['observation.state.master_left_arm_position'].numpy()
        observation_state_master_left_gripper_position = online_buffer[temp_idx]['observation.state.master_left_gripper_position'].numpy()
        observation_state_master_right_arm_position = online_buffer[temp_idx]['observation.state.master_right_arm_position'].numpy()
        observation_state_master_right_gripper_position = online_buffer[temp_idx]['observation.state.master_right_gripper_position'].numpy()
        
        action_left_arm_position = online_buffer[temp_idx]['action.left_arm_position'].numpy()
        action_left_gripper_position = online_buffer[temp_idx]['action.left_gripper_position'].numpy()
        action_right_arm_position = online_buffer[temp_idx]['action.right_arm_position'].numpy()
        action_right_gripper_position = online_buffer[temp_idx]['action.right_gripper_position'].numpy()
        
        # Read images from the three cameras (already resized to 336x336 in online_buffer)
        base_camera = online_buffer[temp_idx]['observation.image.base_camera'].numpy()
        left_camera = online_buffer[temp_idx]['observation.image.left_wrist_camera'].numpy()
        right_camera = online_buffer[temp_idx]['observation.image.right_wrist_camera'].numpy()

        step_data = {
            # Main camera images
            "observation.image.image": base_camera,
            "observation.image.left": left_camera,
            "observation.image.right": right_camera,
            # puppet state
            "observation.state.puppet_left_arm_position": observation_state_puppet_left_arm_position,
            "observation.state.puppet_left_gripper_position": observation_state_puppet_left_gripper_position,
            "observation.state.puppet_right_arm_position": observation_state_puppet_right_arm_position,
            "observation.state.puppet_right_gripper_position": observation_state_puppet_right_gripper_position,
            # master state
            "observation.state.master_left_arm_position": observation_state_master_left_arm_position,
            "observation.state.master_left_gripper_position": observation_state_master_left_gripper_position,
            "observation.state.master_right_arm_position": observation_state_master_right_arm_position,
            "observation.state.master_right_gripper_position": observation_state_master_right_gripper_position,
            # action
            "action.left_arm_position": action_left_arm_position,
            "action.left_gripper_position": action_left_gripper_position,
            "action.right_arm_position": action_right_arm_position,
            "action.right_gripper_position": action_right_gripper_position,
            # raw human_intervention value
            "_human_intervention": online_buffer[temp_idx]['intervention'].numpy(),
        }
        trajectory.append(step_data)
        temp_idx += 1
        traj_len += 1
    
    # Use the y_presses field to determine trajectory success/failure
    # y_presses == 1 means the user pressed 'y' (success)
    # y_presses == 0 means the user pressed 'n' or timed out (failure)
    y_presses_value = online_buffer[temp_idx - 1]['y_presses'].numpy()
    
    human_intervention = np.array(
        [step["_human_intervention"] for step in trajectory], dtype=np.float32
    ).reshape(traj_len, -1)

    # Compute reward and discounted return
    if y_presses_value == 1:
        # Successful trajectory
        rewards = np.zeros((traj_len, 1))
        rewards[-3:] = 1.0
        value_reward = np.ones((traj_len, 1)) * -1
        value_reward[-3:] = 0
    else:
        # Failed trajectory
        rewards = np.zeros((traj_len, 1))
        value_reward = np.ones((traj_len, 1)) * -1
        value_reward[-3:] = -100

    # Compute discounted return
    discount = 1.0
    discounted_return = np.zeros_like(value_reward)
    discounted_return[-1] = value_reward[-1]
    for i in range(traj_len - 2, -1, -1):
        discounted_return[i] = value_reward[i] + discount * discounted_return[i + 1]
    
    # Reward mask
    reward_mask = np.ones((traj_len, 1), dtype=np.bool_)
    reward_mask[-1:] = False

    # Add reward-related fields to each step
    for idx, step_data in enumerate(trajectory):
        step_data.pop("_human_intervention", None)
        step_data["reward"] = rewards[idx].astype(np.float32)
        step_data["human_intervention"] = human_intervention[idx].astype(np.float32)
        step_data["reward_mask"] = reward_mask[idx].astype(np.bool_)
        step_data["value_reward"] = value_reward[idx].astype(np.float32)
        step_data["discounted_value_return"] = discounted_return[idx].astype(np.float32)
    
    return trajectory, temp_idx, y_presses_value


def process_subdir(subdir, sync_data_dir, output_dir, default_task_name, task_description, print_lock):
    """
    Convert a single subdirectory; intended to be called from multiple threads.
    Each subdirectory has its own independent online_buffer and LeRobotDataset, with no interference between them.

    Returns: (subdir name, number of trajectories converted), or (subdir name, 0) on error
    """
    subdir_path = os.path.join(sync_data_dir, subdir)

    # Extract the task name from the folder name (assumes format task_name_YYYYMMDD_HHMMSS)
    if '_' in subdir:
        parts = subdir.split('_')
        task_name = '_'.join(parts[:-2]) if len(parts) >= 3 else default_task_name
    else:
        task_name = default_task_name

    # Load the online_buffer
    try:
        online_buffer = RealWorldOnlineBuffer(
            temp_dir=subdir_path,
            data_spec=X_HUMANOID_Online_DATA_SPEC,
            buffer_capacity=500000,
            fps=30
        )
        traj_num = online_buffer.traj_num
        with print_lock:
            print(f"\n{'='*60}")
            print(f"[Thread] Processing folder: {subdir}")
            print(f"  Trajectory count: {traj_num}  Total frames: {len(online_buffer)}")
            print(f"{'='*60}")
    except Exception as e:
        with print_lock:
            print(f"  [Error] Failed to load data folder {subdir}: {e}")
        return subdir, 0

    # Create the output dataset (each subdirectory has its own independent path, no conflicts)
    output_path = os.path.join(output_dir, subdir)
    dataset = LeRobotDataset.create(
        features=X_HUMANOID_REAL_WORLD_FEATURES,
        root=output_path,
        repo_id=f"x_humanoid/{task_name}",
        fps=30,
        robot_type="x_humanoid",
    )

    # Process all trajectories in this subdirectory sequentially (frame order must be preserved)
    st_idx = 0
    for traj_index in range(traj_num):
        trajectory, temp_idx, is_success = load_trajectory(traj_index, online_buffer, st_idx)

        task_desc = task_description
        for single_step in trajectory:
            single_step["task"] = task_desc
            dataset.add_frame(single_step)

        with print_lock:
            status = 'SUCCESS' if is_success == 1 else 'FAILED'
            print(f"  [{subdir}] Trajectory {traj_index}/{traj_num-1} - {status}")

        dataset.save_episode()
        st_idx = temp_idx

    with print_lock:
        print(f"  ✓ [{subdir}] Done, {traj_num} trajectories total, output: {output_path}")

    return subdir, traj_num


if __name__ == "__main__":
    # Base path configuration
    sync_data_dir = "/home/wink/Desktop/code/hierarchical_rl_in_real_world/tmp/best_rtc_data_rollout"
    output_dir = "/home/wink/Desktop/code/hierarchical_rl_in_real_world/tmp/test_multi"

    # Task name and description (extracted from folder name or use the default)
    default_task_name = "grasp_buttons_from_the_conveyor_belt_and_sort_by_color_into_different_plates"
    task_description = "grasp buttons from the conveyor belt and sort by color into different plates"

    # Number of parallel threads (adjust based on CPU/IO resources, 4-8 recommended)
    max_workers = 4

    if not os.path.exists(sync_data_dir):
        print(f"Error: directory does not exist {sync_data_dir}")
        exit(1)

    subdirs = sorted([
        d for d in os.listdir(sync_data_dir)
        if os.path.isdir(os.path.join(sync_data_dir, d))
    ])

    if not subdirs:
        print(f"Warning: no subfolders found in {sync_data_dir}")
        exit(1)

    print(f"Found {len(subdirs)} data folders, processing in parallel with {max_workers} threads...\n")

    print_lock = threading.Lock()
    total_traj_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_subdir,
                subdir, sync_data_dir, output_dir,
                default_task_name, task_description, print_lock
            ): subdir
            for subdir in subdirs
        }

        for future in as_completed(futures):
            subdir_name = futures[future]
            try:
                _, traj_num = future.result()
                total_traj_count += traj_num
            except Exception as e:
                with print_lock:
                    print(f"  [Uncaught exception] {subdir_name}: {e}")

    print(f"\n{'='*60}")
    print(f"All done! Processed {len(subdirs)} folders, {total_traj_count} trajectories total")
    print(f"{'='*60}")
