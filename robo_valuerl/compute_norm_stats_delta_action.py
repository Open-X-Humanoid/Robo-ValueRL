import pandas as pd
import os
import numpy as np
import json
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor

# 1. Define data structures
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    q01: Optional[np.ndarray] = None
    q99: Optional[np.ndarray] = None

    def to_dict(self):
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "min": self.min.tolist() if self.min is not None else None,
            "max": self.max.tolist() if self.max is not None else None,
            "q01": self.q01.tolist() if self.q01 is not None else None,
            "q99": self.q99.tolist() if self.q99 is not None else None,
        }

def extract_raw_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    def get_val(key, is_gripper=False):
        val = np.array(data[key].tolist())
        if is_gripper and val.ndim == 1:
            val = val[:, np.newaxis]
        return val

    s_l_arm = get_val('observation.state.puppet_left_arm_position')
    s_l_grip = get_val('observation.state.puppet_left_gripper_position', True)
    s_r_arm = get_val('observation.state.puppet_right_arm_position')
    s_r_grip = get_val('observation.state.puppet_right_gripper_position', True)

    l_arm_dim = s_l_arm.shape[1]
    l_grip_idx = l_arm_dim
    r_grip_idx = l_arm_dim + 1 + s_r_arm.shape[1]
    gripper_indices = [l_grip_idx, r_grip_idx]

    state = np.concatenate([s_l_arm, s_l_grip, s_r_arm, s_r_grip], axis=-1)

    a_l_arm = get_val('action.left_arm_position')
    a_l_grip = get_val('action.left_gripper_position', True)
    a_r_arm = get_val('action.right_arm_position')
    a_r_grip = get_val('action.right_gripper_position', True)
    action = np.concatenate([a_l_arm, a_l_grip, a_r_arm, a_r_grip], axis=-1)
    
    return state, action, gripper_indices

def process_single_file(args) -> Dict:
    """
    Process a single file in parallel, computing local accumulated sums.
    """
    file_path, chunk_size, stride = args
    try:
        df = pd.read_parquet(file_path)
        state, action, g_indices = extract_raw_data(df)
        
        num_frames = state.shape[0]
        # Use strided sampling to greatly speed things up without losing statistical properties
        indices = range(0, num_frames, stride)
        
        local_deltas = []
        local_states = []
        
        for t in indices:
            curr_state = state[t]
            end_idx = min(t + chunk_size, num_frames)
            # Compute the Delta for the current chunk
            delta_chunk = action[t:end_idx] - curr_state
            local_deltas.append(delta_chunk)
            local_states.append(curr_state)
            
        deltas_all = np.concatenate(local_deltas, axis=0)
        states_all = np.array(local_states)
        
        return {
            "s_sum": states_all.sum(axis=0),
            "s_sum_sq": (states_all**2).sum(axis=0),
            "s_min": states_all.min(axis=0),
            "s_max": states_all.max(axis=0),
            "s_count": states_all.shape[0],
            "d_sum": deltas_all.sum(axis=0),
            "d_sum_sq": (deltas_all**2).sum(axis=0),
            "d_min": deltas_all.min(axis=0),
            "d_max": deltas_all.max(axis=0),
            "d_count": deltas_all.shape[0],
            "g_indices": g_indices
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def finalize_stats(total_sum, total_sum_sq, total_min, total_max, count, gripper_indices, target_dim=32) -> NormStats:
    """
    Compute final mean, variance, and min/max from accumulated sums, and apply the gripper 0/1 constraint.
    """
    mean = total_sum / count
    # Var = E[X^2] - (E[X])^2
    var = (total_sum_sq / count) - (mean**2)
    std = np.sqrt(np.clip(var, a_min=1e-7, a_max=None))

    # Special handling for the gripper
    for idx in gripper_indices:
        if idx < mean.shape[0]:
            mean[idx] = 0.0
            std[idx] = 1.0

    def pad_or_clip(arr, fill_val):
        curr_d = arr.shape[0]
        if curr_d < target_dim:
            return np.concatenate([arr, np.full(target_dim - curr_d, fill_val)])
        return arr[:target_dim]

    return NormStats(
        mean=pad_or_clip(mean, 0.0),
        std=pad_or_clip(std, 1.0),
        min=pad_or_clip(total_min, 0.0),
        max=pad_or_clip(total_max, 0.0)
    )

if __name__ == "__main__":
    data_dir = "/mnt/dataset/toago6/block_x_humanoid_data/need_finegrained_filtered_output"
    # data_list = [
        # 'x_humanoid_lerobot_data_with_quality_30_01_04',
        # 'x_humanoid_lerobot_data_with_quality_16_01_06',
        # 'x_humanoid_lerobot_data_with_quality_41_01_13', 
        # 'x_humanoid_lerobot_data_with_quality_16_12_30', 
        # 'x_humanoid_lerobot_data_with_quality_41_01_12', 
        # 'x_humanoid_lerobot_data_with_quality_41_01_08', 
        # 'x_humanoid_lerobot_data_without_discount_23', 
        # 'x_humanoid_lerobot_data_with_quality_16_12_27_suboptimal', 
        # 'x_humanoid_lerobot_data_with_quality_41_01_09', 
        # 'x_humanoid_lerobot_data_with_quality_16_01_07', 
        # 'x_humanoid_lerobot_data_without_discount_26', 
        # 'x_humanoid_lerobot_data_with_quality_suboptimal_30_01_05',
        # 'x_humanoid_lerobot_data_without_discount_24', 
        # 'x_humanoid_lerobot_data_with_quality_30_01_07', 
        # 'x_humanoid_lerobot_data_without_discount_with_quality_27', 
        # 'x_humanoid_lerobot_data_with_quality_30_01_06',
        # 'x_humanoid_lerobot_data_with_quality_suboptimal_30_12_31', 
        # 'x_humanoid_lerobot_data_with_quality_16_01_13',
        # 'x_humanoid_lerobot_data_with_quality_16_01_05',
        # 'x_humanoid_lerobot_data_with_quality_suboptimal_30_01_06',
        # 'x_humanoid_lerobot_data_with_quality_suboptimal_30_12_30', 
        # 'x_humanoid_lerobot_data_with_quality_16_12_27', 
        # 'x_humanoid_lerobot_data_without_discount_22',
        # 'x_humanoid_lerobot_data_without_discount_20', 
        # '16_x_humanoid_lerobot_data_without_discount_24', 
        # 'x_humanoid_lerobot_data_with_quality_16_01_08',
        # 'x_humanoid_lerobot_data_with_quality_16_12_31',
        # 'x_humanoid_lerobot_data_with_quality_30_12_29', 
        # 'x_humanoid_lerobot_data_with_quality_30_12_30', 
        # 'x_humanoid_lerobot_data_with_quality_suboptimal_30_01_04', 
        # 'x_humanoid_lerobot_data_with_quality_16_01_09', 
        # 'x_humanoid_lerobot_data_with_quality_16_01_04'
    # ]
    data_list = os.listdir(data_dir)
    
    chunk_size = 50
    stride = 5  # Key optimization: sample one window every 10 frames; statistics remain essentially unbiased while speed improves 10x
    target_dim = 32
    
    file_tasks = []
    for data_name in data_list:
        target_path = os.path.join(data_dir, data_name, "data")
        if not os.path.exists(target_path): continue
        for root, _, files in os.walk(target_path):
            for file in files:
                if file.endswith(".parquet"):
                    file_tasks.append((os.path.join(root, file), chunk_size, stride))

    print(f"Starting parallel computation, total files: {len(file_tasks)}, sampling stride: {stride}")

    # Global accumulators
    g_s_sum, g_s_sum_sq, g_s_count = 0, 0, 0
    g_d_sum, g_d_sum_sq, g_d_count = 0, 0, 0
    g_s_min, g_s_max = None, None
    g_d_min, g_d_max = None, None
    gripper_indices = []

    # Use a multiprocessing pool
    with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
        results = list(executor.map(process_single_file, file_tasks))

    for res in results:
        if res:
            g_s_sum += res["s_sum"]
            g_s_sum_sq += res["s_sum_sq"]
            g_s_count += res["s_count"]
            g_d_sum += res["d_sum"]
            g_d_sum_sq += res["d_sum_sq"]
            g_d_count += res["d_count"]

            # Merge minimum values
            if g_s_min is None:
                g_s_min = res["s_min"]
            else:
                g_s_min = np.minimum(g_s_min, res["s_min"])

            # Merge maximum values
            if g_s_max is None:
                g_s_max = res["s_max"]
            else:
                g_s_max = np.maximum(g_s_max, res["s_max"])

            # Merge delta minimum values
            if g_d_min is None:
                g_d_min = res["d_min"]
            else:
                g_d_min = np.minimum(g_d_min, res["d_min"])

            # Merge delta maximum values
            if g_d_max is None:
                g_d_max = res["d_max"]
            else:
                g_d_max = np.maximum(g_d_max, res["d_max"])

            gripper_indices = res["g_indices"]

    if g_s_count > 0:
        print("Computing final statistics and applying gripper 0/1 constraint...")
        state_stats = finalize_stats(g_s_sum, g_s_sum_sq, g_s_min, g_s_max, g_s_count, gripper_indices, target_dim)
        delta_stats = finalize_stats(g_d_sum, g_d_sum_sq, g_d_min, g_d_max, g_d_count, gripper_indices, target_dim)

        final_output = {
            "state": state_stats.to_dict(),
            "actions": delta_stats.to_dict()
        }

        save_file = "with_max_min_humanoid_delta_fast_stats.json"
        with open(save_file, "w") as f:
            json.dump(final_output, f, indent=4)
        
        print(f"Done! Data has been saved to {save_file}")
        print(f"Total samples processed: State={g_s_count}, Delta_Steps={g_d_count}")