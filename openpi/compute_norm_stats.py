import pandas as pd
import os
import numpy as np
import pydantic
import numpydantic
import json
from typing import Optional

# 1. Define data structures
@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: Optional[numpydantic.NDArray] = None
    q99: Optional[numpydantic.NDArray] = None

    def to_dict(self):
        """Convert numpy arrays to lists for JSON serialization"""
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "q01": self.q01.tolist() if self.q01 is not None else None,
            "q99": self.q99.tolist() if self.q99 is not None else None,
        }

def get_state_info(data: pd.DataFrame):
    """
    Extract and concatenate state and action from the data frame
    """
    def get_val(key, is_gripper=False):
        # Convert pandas column (possibly list format) to numpy array (N, D)
        val = np.array(data[key].tolist())
        # If it's 1D gripper data, expand to (N, 1)
        if is_gripper and val.ndim == 1:
            val = val[:, np.newaxis]
        return val

    # State concatenation (left arm pose + left gripper + right arm pose + right gripper)
    s_l_arm = get_val('observation.state.puppet_left_arm_position')
    s_l_grip = get_val('observation.state.puppet_left_gripper_position', True)
    s_r_arm = get_val('observation.state.puppet_right_arm_position')
    s_r_grip = get_val('observation.state.puppet_right_gripper_position', True)
    state = np.concatenate([s_l_arm, s_l_grip, s_r_arm, s_r_grip], axis=-1)

    # Action concatenation
    a_l_arm = get_val('action.left_arm_position')
    a_l_grip = get_val('action.left_gripper_position', True)
    a_r_arm = get_val('action.right_arm_position')
    a_r_grip = get_val('action.right_gripper_position', True)
    action = np.concatenate([a_l_arm, a_l_grip, a_r_arm, a_r_grip], axis=-1)
    
    return state, action

def calculate_stats(data_list: list, target_dim: int = 32) -> NormStats:
    """
    Compute statistics and pad dimensions: for dims short of 32, pad mean with 0 and std with 1
    """
    # Concatenate all blocks into one large array (Total_Frames, D)
    full_data = np.concatenate(data_list, axis=0)

    # Basic statistics computation
    curr_mean = np.mean(full_data, axis=0)
    curr_std = np.std(full_data, axis=0)
    curr_std = np.clip(curr_std, a_min=1e-6, a_max=None) # Avoid division by zero
    curr_q01 = np.quantile(full_data, 0.01, axis=0)
    curr_q99 = np.quantile(full_data, 0.99, axis=0)

    curr_dim = curr_mean.shape[0]

    # Dimension padding logic
    if curr_dim < target_dim:
        pad_size = target_dim - curr_dim
        # Pad mean with 0, std with 1, quantiles with 0
        mean = np.concatenate([curr_mean, np.zeros(pad_size)])
        std = np.concatenate([curr_std, np.ones(pad_size)])
        q01 = np.concatenate([curr_q01, np.zeros(pad_size)])
        q99 = np.concatenate([curr_q99, np.zeros(pad_size)])
    else:
        # Truncate if exceeding 32 dims
        mean, std, q01, q99 = curr_mean[:target_dim], curr_std[:target_dim], curr_q01[:target_dim], curr_q99[:target_dim]

    return NormStats(mean=mean, std=std, q01=q01, q99=q99)

if __name__ == "__main__":
    data_dir = "/mnt/dataset/toago6/block_x_humanoid_data/filtered_version"
    data_list = [
        "x_humanoid_lerobot_data_without_discount_20",
        "x_humanoid_lerobot_data_without_discount_22",
        "x_humanoid_lerobot_data_without_discount_23",
        "x_humanoid_lerobot_data_without_discount_24",
        "x_humanoid_lerobot_data_without_discount_26",
        "x_humanoid_lerobot_data_with_quality_16_12_27",
        "x_humanoid_lerobot_data_without_discount_with_quality_27"
    ]
    
    all_states = []
    all_actions = []
    total_frames = 0  # Count total frames

    for data_name in data_list:
        target_path = os.path.join(data_dir, data_name, "data")
        if not os.path.exists(target_path):
            print(f"Skipping nonexistent path: {target_path}")
            continue

        print(f"Reading dataset: {data_name}")
        for root, _, files in os.walk(target_path):
            for file in files:
                if file.endswith(".parquet"):
                    file_path = os.path.join(root, file)
                    df = pd.read_parquet(file_path)

                    # Accumulate frame count (number of rows)
                    total_frames += len(df)

                    state, action = get_state_info(df)
                    all_states.append(state)
                    all_actions.append(action)

    # Compute statistics uniformly
    if all_states:
        print("\nComputing statistics and padding dimensions...")
        state_norm = calculate_stats(all_states, target_dim=32)
        action_norm = calculate_stats(all_actions, target_dim=32)

        # Build final output dictionary
        final_output = {
            "state": state_norm.to_dict(),
            "actions": action_norm.to_dict(),
            # "metadata": {
            #     "total_frames": int(total_frames),
            #     "dimension": 32,
            #     "info": "Padded with mean=0 and std=1 for dimensions > original"
            # }
        }

        # Save as JSON file
        save_file = "dataset_stats.json"
        with open(save_file, "w") as f:
            json.dump(final_output, f, indent=4)

        print("-" * 40)
        print(f"Computation complete!")
        print(f"Total frames processed (Total Frames): {total_frames}")
        print(f"State dimension: {state_norm.mean.shape[0]}")
        print(f"Action dimension: {action_norm.mean.shape[0]}")
        print(f"Results saved to: {save_file}")
    else:
        print("Error: No valid data found.")