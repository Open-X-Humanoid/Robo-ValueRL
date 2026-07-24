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
        val = np.array(data[key].tolist())
        if is_gripper and val.ndim == 1:
            val = val[:, np.newaxis]
        return val

    # State concatenation
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
    Compute statistics and pad dimensions
    """
    # Concatenate all state and action blocks into one huge array (N_frames * 2, D)
    full_data = np.concatenate(data_list, axis=0)
    
    curr_mean = np.mean(full_data, axis=0)
    curr_std = np.std(full_data, axis=0)
    curr_std = np.clip(curr_std, a_min=1e-6, a_max=None) 
    curr_q01 = np.quantile(full_data, 0.01, axis=0)
    curr_q99 = np.quantile(full_data, 0.99, axis=0)
    
    curr_dim = curr_mean.shape[0]
    
    if curr_dim < target_dim:
        pad_size = target_dim - curr_dim
        mean = np.concatenate([curr_mean, np.zeros(pad_size)])
        std = np.concatenate([curr_std, np.ones(pad_size)])
        q01 = np.concatenate([curr_q01, np.zeros(pad_size)])
        q99 = np.concatenate([curr_q99, np.zeros(pad_size)])
    else:
        mean, std, q01, q99 = curr_mean[:target_dim], curr_std[:target_dim], curr_q01[:target_dim], curr_q99[:target_dim]

    return NormStats(mean=mean, std=std, q01=q01, q99=q99)

if __name__ == "__main__":
    data_dir = "/mnt/dataset/toago6/block_x_humanoid_data/filtered_version"
    data_list = [
                      "x_humanoid_lerobot_data_without_discount_24",
                      "x_humanoid_lerobot_data_without_discount_26",
                      "x_humanoid_lerobot_data_with_quality_16_12_27",
                      "x_humanoid_lerobot_data_without_discount_with_quality_27",
                      "x_humanoid_lerobot_data_with_quality_30_01_04",
                      "x_humanoid_lerobot_data_with_quality_30_01_05",
                      "x_humanoid_lerobot_data_with_quality_30_01_06",
                      "x_humanoid_lerobot_data_with_quality_30_12_29",
                      "x_humanoid_lerobot_data_with_quality_30_12_30"
                ]
    
    # Modified: use a single unified list to store all data
    all_combined_data = []
    total_frames = 0
    
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
                    
                    total_frames += len(df)
                    
                    state, action = get_state_info(df)
                    # Modified: put both state and action into the same list
                    all_combined_data.append(state)
                    all_combined_data.append(action)

    if all_combined_data:
        print(f"\nComputing statistics uniformly for {total_frames * 2} vectors and padding dimensions...")

        # Compute once, uniformly
        unified_norm = calculate_stats(all_combined_data, target_dim=32)
        unified_stats_dict = unified_norm.to_dict()

        # Modified: state and actions use identical statistics
        final_output = {
            "state": unified_stats_dict,
            "actions": unified_stats_dict
        }

        save_file = "30_24_06_dataset_stats.json"
        with open(save_file, "w") as f:
            json.dump(final_output, f, indent=4)
        
        print("-" * 40)
        print(f"Computation complete!")
        print(f"Total raw frames processed: {total_frames}")
        print(f"Combined sample count: {total_frames * 2}")
        print(f"Results saved to: {save_file}")
    else:
        print("Error: No valid data found.")