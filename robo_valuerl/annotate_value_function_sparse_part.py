import os
import torch
import numpy as np
import pandas as pd
import av
import json
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
# OpenPi-related imports
import openpi.training.config as _config
import transforms.transforms as _transforms
from openpi.models.hl_gauss_distribution import HLGaussDistribution
from transforms.tokenizer import PaligemmaTokenizer
from agents.openpi_x_humanoid_critic import OpenPi_X_Humanoid_Critic

# --- 1. Core utility functions ---

def read_video_frames_pyav(video_path):
    frames = []
    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format='rgb24'))
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
    return frames

# --- 2. Interpolation and smoothing functions ---

def linear_interpolate(key_data, data_len):
    """Linear interpolation: linearly interpolate values between key frames"""
    key_indices = sorted(key_data.keys())
    key_values = [key_data[idx] for idx in key_indices]

    if len(key_indices) < 2:
        return np.full(data_len, key_values[0] if key_values else 0).tolist()

    all_indices = np.arange(data_len)
    interpolated = np.interp(all_indices, key_indices, key_values)

    return interpolated.tolist()


def cubic_spline_interpolate(key_data, data_len, bc_type='natural'):
    """
    Cubic spline interpolation: smooth interpolation using Cubic Spline

    Args:
        key_data: {frame_idx: value} dict of key frames
        data_len: total number of frames
        bc_type: boundary condition type
    """
    key_indices = sorted(key_data.keys())
    key_values = [key_data[idx] for idx in key_indices]

    if len(key_indices) < 2:
        return np.full(data_len, key_values[0] if key_values else 0).tolist()

    if len(key_indices) < 4:
        # Use linear interpolation when there are too few key points
        return linear_interpolate(key_data, data_len)

    cs = CubicSpline(key_indices, key_values, bc_type=bc_type)
    all_indices = np.arange(data_len)
    interpolated = cs(all_indices)

    return interpolated.tolist()


def savitzky_golay_smooth(data, window_length=21, polyorder=3):
    """
    Savitzky-Golay filtering: smooth the time series data

    Args:
        data: time series data
        window_length: window length (must be odd and greater than polyorder)
        polyorder: polynomial order
    """
    data = np.array(data)
    data_len = len(data)

    # Skip smoothing when the data is too short
    if data_len < 3:
        return data.tolist()

    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1

    # window_length cannot exceed the data length
    window_length = min(window_length, data_len)
    if window_length < 3:
        window_length = 3

    # polyorder must be less than window_length and non-negative
    polyorder = min(polyorder, window_length - 1)
    if polyorder < 1:
        polyorder = 1

    pad_width = window_length // 2
    data_padded = np.pad(data, pad_width, mode='edge')

    smoothed = savgol_filter(data_padded, window_length, polyorder)

    return smoothed[pad_width:pad_width + data_len].tolist()


def interpolate_and_smooth(key_data, data_len, method='linear', smooth=False,
                          smooth_window=21, smooth_polyorder=3, bc_type='natural'):
    """
    Unified interpolation and smoothing interface

    Args:
        key_data: {frame_idx: value} dict of key frames
        data_len: total number of frames
        method: interpolation method ('linear' or 'cubic')
        smooth: whether to apply Savitzky-Golay smoothing
    """
    if method == 'cubic':
        result = cubic_spline_interpolate(key_data, data_len, bc_type=bc_type)
    else:
        result = linear_interpolate(key_data, data_len)

    if smooth:
        result = savitzky_golay_smooth(result, smooth_window, smooth_polyorder)

    return result


# --- 3. Task map loading function ---

def load_task_map(data_path):
    """Build task_index -> task_text mapping from meta/tasks.jsonl

    Args:
        data_path: parquet file path, format .../lerobot/<dataset>/data/chunk-XXX/episode_XXXXXX.parquet

    Returns:
        task_map: {task_index: task_text} dict
    """
    # tasks.jsonl location: .../lerobot/<dataset>/meta/tasks.jsonl
    meta_path = Path(data_path).parent.parent.parent / "meta" / "tasks.jsonl"
    task_map = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    task_map[obj["task_index"]] = obj["task"]
    return task_map


# --- 4. Inference and annotation function ---

def process_and_annotate(data_path, v_paths, agent, v_hl, t_hl, idx, config,
                         stride=10, interpolation_method='linear', smooth=False,
                         smooth_window=21, smooth_polyorder=3, bc_type='natural'):
    """
    Run a prediction every `stride` frames, then interpolate and smooth the intermediate frames

    Args:
        data_path: parquet file path
        v_paths: dict of video file paths
        agent: prediction model
        v_hl: Gaussian distribution decoder for value
        t_hl: Gaussian distribution decoder for time
        idx: trajectory index
        stride: sampling stride (how many frames between predictions)
        interpolation_method: interpolation method ('linear' or 'cubic')
        smooth: whether to apply smoothing
    """
    data = pd.read_parquet(data_path)
    task_map = load_task_map(data_path)
    frames_base = read_video_frames_pyav(v_paths['base'])
    frames_left = read_video_frames_pyav(v_paths['left'])
    frames_right = read_video_frames_pyav(v_paths['right'])

    if not frames_base:
        print(f"Warning: No frames found for {v_paths['base']}")
        return

    data_len = min(len(data), len(frames_base))

    # History frame offset (read from config)
    history_offset = config.history_length

    # --- Stage A: Inference stage (only run inference on key frames) ---
    print(f"--> [Trajectory {idx}] Predicting key frames (stride={stride})...")

    # Store inference results for key frames
    key_vals = {}      # {frame_idx: predicted_value}
    key_rtgs = {}      # {frame_idx: predicted_rtg}
    key_times = {}     # {frame_idx: predicted_time}

    key_indices = list(range(0, data_len, stride))
    # Ensure the last frame is also included
    if key_indices[-1] != data_len - 1:
        key_indices.append(data_len - 1)

    for i in tqdm(key_indices, desc="Inference"):
        # Build the state vector (use np.atleast_1d to ensure every field is an array)
        state = np.concatenate([
            np.atleast_1d(data['observation.state.puppet_left_arm_position'][i]),
            np.atleast_1d(data['observation.state.puppet_left_gripper_position'][i]),
            np.atleast_1d(data['observation.state.puppet_right_arm_position'][i]),
            np.atleast_1d(data['observation.state.puppet_right_gripper_position'][i]),
        ])
        # Build the action vector
        action = np.concatenate([
            np.atleast_1d(data['action.left_arm_position'][i]),
            np.atleast_1d(data['action.left_gripper_position'][i]),
            np.atleast_1d(data['action.right_arm_position'][i]),
            np.atleast_1d(data['action.right_gripper_position'][i]),
        ])

        history_idx = max(0, i - history_offset)

        batch = {
            "image": {
                "base_0_rgb": np.expand_dims(frames_base[i], 0),
                "left_wrist_0_rgb": np.expand_dims(frames_left[i], 0),
                "right_wrist_0_rgb": np.expand_dims(frames_right[i], 0),
            },
            "history_image": np.expand_dims(frames_base[history_idx], 0),
            "state": torch.from_numpy(state).float().unsqueeze(0),
            "actions": torch.from_numpy(action).float().unsqueeze(0),
            "prompt": task_map.get(int(data['task_index'][i]), ""),
            "image_mask": {
                "base_0_rgb": torch.tensor([True]),
                "left_wrist_0_rgb": torch.tensor([True]),
                "right_wrist_0_rgb": torch.tensor([True]),
            },
            "frame_index": torch.tensor([i])
        }

        with torch.no_grad():
            preds = agent.predict_value(batch)
            v_soft, r_soft, t_soft = preds['value'].cpu(), preds['rtg'].cpu(), preds['remain_time'].cpu()

            key_vals[i] = v_hl.batch_decode(v_soft).item()
            key_times[i] = t_hl.batch_decode(t_soft).item()
            key_rtgs[i] = torch.argmax(r_soft).item() * 50

    # --- Stage B: Interpolate to generate predicted values for all frames ---
    smooth_str = " + Savitzky-Golay smoothing" if smooth else ""
    print(f"--> [Trajectory {idx}] Interpolating {len(key_vals)} key frames to {data_len} frames ({interpolation_method}{smooth_str})...")

    p_vals = interpolate_and_smooth(key_vals, data_len, method=interpolation_method, smooth=smooth,
                                    smooth_window=smooth_window, smooth_polyorder=smooth_polyorder, bc_type=bc_type)
    p_times = interpolate_and_smooth(key_times, data_len, method=interpolation_method, smooth=smooth,
                                     smooth_window=smooth_window, smooth_polyorder=smooth_polyorder, bc_type=bc_type)
    p_rtgs = interpolate_and_smooth(key_rtgs, data_len, method=interpolation_method, smooth=smooth,
                                    smooth_window=smooth_window, smooth_polyorder=smooth_polyorder, bc_type=bc_type)

    # --- Stage C: Save to parquet file ---
    data["predict_value_function"] = p_vals
    data["predict_remain_time"] = p_times
    data["predict_return_to_go"] = p_rtgs

    data.to_parquet(data_path)
    print(f"--> [Trajectory {idx}] Saved predictions to {data_path}")


# --- 5. Configuration parameter class ---

class InterpolationConfig:
    """Interpolation and smoothing configuration"""
    def __init__(self,
                 stride: int = 10,
                 interpolation_method: str = 'cubic',
                 smooth: bool = True,
                 smooth_window: int = 21,
                 smooth_polyorder: int = 3,
                 bc_type: str = 'natural'):
        self.stride = stride
        self.interpolation_method = interpolation_method
        self.smooth = smooth
        self.smooth_window = smooth_window
        self.smooth_polyorder = smooth_polyorder
        self.bc_type = bc_type

    def __str__(self):
        smooth_str = f" + SG(window={self.smooth_window}, poly={self.smooth_polyorder})" if self.smooth else ""
        return f"stride={self.stride}, {self.interpolation_method}_interpolation{smooth_str}"


# --- 6. Main entry point ---

def main(interp_config: InterpolationConfig = None):
    """
    Args:
        interp_config: interpolation configuration; uses the default configuration if None
    """
    if interp_config is None:
        interp_config = InterpolationConfig()  # default configuration: stride=10, cubic+smooth

    print(f"=== Interpolation Config: {interp_config} ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = _config.cli()

    v_hl = HLGaussDistribution(torch.linspace(0, 400, 256), sigma=3.0)
    t_hl = HLGaussDistribution(torch.linspace(0, 5000, 256), sigma=3.0)

    transforms = _transforms.compose([
        _transforms.InjectDefaultPrompt(prompt="separate the blocks and sort by color into different plates"),
        _transforms.TokenizePrompt(PaligemmaTokenizer(max_len=48)),
        _transforms.ResizeImages(256, 256),
        _transforms.PadStatesAndActions(config.model.action_dim)
    ])

    image_transforms = A.Compose([
        A.CenterCrop(height=224, width=224, p=1.0),
    ])
    agent = OpenPi_X_Humanoid_Critic(config, chunk_size=10, data_transforms=transforms, image_transforms=image_transforms).to(device)
    agent.eval()

    data_dir_root = config.annotation_dir_root

    all_data_list = sorted(os.listdir(data_dir_root))
    annotate_total = config.annotate_total
    annotate_count = config.annotate_count
    
    chunk_size = (len(all_data_list) + annotate_total - 1) // annotate_total
    cur_data_list = all_data_list[annotate_count * chunk_size:(annotate_count + 1) * chunk_size]

    failed_episodes = []  # Track failed episodes

    for data_dir in cur_data_list:
        if os.path.isdir(os.path.join(data_dir_root, data_dir)):
            data_dir = os.path.join(data_dir_root, data_dir)
            parquet_dir = os.path.join(data_dir, "data", "chunk-000")
            if not os.path.exists(parquet_dir):
                continue
            if not os.path.isdir(parquet_dir):
                continue

            files = sorted([f for f in os.listdir(parquet_dir) if f.endswith(".parquet")])

            for i, name in enumerate(files):
                p_path = os.path.join(parquet_dir, name)

                print("this is the name of the file: ", name)
                v_paths = {
                    'base': os.path.join(data_dir, "videos/chunk-000/observation.image.image", name.replace(".parquet", ".mp4")),
                    'left': os.path.join(data_dir, "videos/chunk-000/observation.image.left", name.replace(".parquet", ".mp4")),
                    'right': os.path.join(data_dir, "videos/chunk-000/observation.image.right", name.replace(".parquet", ".mp4"))
                }
                try:
                    process_and_annotate(p_path, v_paths, agent, v_hl, t_hl, i, config,
                                        stride=interp_config.stride,
                                        interpolation_method=interp_config.interpolation_method,
                                        smooth=interp_config.smooth,
                                        smooth_window=interp_config.smooth_window,
                                        smooth_polyorder=interp_config.smooth_polyorder,
                                        bc_type=interp_config.bc_type)
                except Exception as e:
                    print(f"[ERROR] Failed to process {p_path}: {e}")
                    failed_episodes.append(p_path)
                    continue

            # fill the meta information
            info_path = os.path.join(data_dir, "meta", "info.json")
            with open(info_path, "r") as f:
                info_data = json.load(f)
            info_data["features"]["predict_value_function"] = {"dtype": "float32", "shape": [1], "names": None}
            info_data["features"]["predict_remain_time"] = {"dtype": "float32", "shape": [1], "names": None}
            info_data["features"]["predict_return_to_go"] = {"dtype": "float32", "shape": [1], "names": None}
            with open(info_path, "w") as f:
                json.dump(info_data, f, indent=4)

    # Final summary
    if failed_episodes:
        print(f"\n{'='*60}")
        print(f"[SUMMARY] {len(failed_episodes)} episode(s) failed:")
        for ep in failed_episodes:
            print(f"  - {ep}")
        print(f"{'='*60}")
    else:
        print("\n[SUMMARY] All episodes processed successfully.")


if __name__ == "__main__":
    main()
