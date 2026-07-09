import os
import cv2
import torch
import numpy as np
import pandas as pd
import av
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

# --- 1. Core utility functions (keep your original algorithm) ---

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

def preprocess_frame_for_display(frame, resize_size=(256, 256), crop_size=(224, 224)):
    """
    Preprocess frame for display: resize to (256,256) then center crop to (224,224)
    Args:
        frame: numpy array (H, W, 3) in RGB format
        resize_size: target size for resizing (height, width)
        crop_size: target size for center crop (height, width)
    Returns:
        preprocessed frame in RGB format
    """
    resize_h, resize_w = resize_size
    crop_h, crop_w = crop_size

    # Step 1: Resize
    resized = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

    # Step 2: Center Crop
    start_h = (resize_h - crop_h) // 2
    start_w = (resize_w - crop_w) // 2
    cropped = resized[start_h:start_h+crop_h, start_w:start_w+crop_w]

    return cropped

def get_plateau_mask(data, min_len=100):
    """Detect segments in the sequence where consecutive identical values exceed min_len (used to turn the Value curve red)"""
    mask = np.zeros(len(data), dtype=bool)
    if len(data) == 0: return mask
    count = 1
    for i in range(1, len(data)):
        if abs(data[i] - data[i-1]) < 1: # original threshold
            count += 1
        else:
            if count >= min_len:
                mask[i-count : i] = True
            count = 1
    if count >= min_len:
        mask[len(data)-count : len(data)] = True
    return mask

def draw_text_with_bg(img, text, pos, font_scale=0.6, color=(255, 255, 255), thickness=1):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness+1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# --- 2. 2x2 dedicated plotting components ---

def draw_trend_panel(panel, data, current_idx, title, unit, color, y_range, plateau_mask=None):
    ph, pw, _ = panel.shape
    pad_l, pad_r, pad_t, pad_b = 65, 30, 45, 40
    cw, ch = pw - pad_l - pad_r, ph - pad_t - pad_b
    
    cv2.rectangle(panel, (2, 2), (pw-2, ph-2), (22, 22, 22), -1)
    v_min, v_max = y_range
    v_span = max(v_max - v_min, 1e-5)

    def to_y(val): return int(pad_t + ch - ((val - v_min) / v_span) * ch)
    def to_x(idx): return int(pad_l + (idx / (len(data)-1)) * cw)

    # grid lines
    for i in range(5):
        y_val = v_min + i * (v_span / 4)
        cv2.line(panel, (pad_l, to_y(y_val)), (pad_l + cw, to_y(y_val)), (40, 40, 40), 1)

    if len(data) > 1:
        for j in range(1, current_idx + 1):
            p1 = (to_x(j-1), to_y(data[j-1]))
            p2 = (to_x(j), to_y(data[j]))
            # Logic: if plateau_mask[j] is True, color this segment red
            draw_color = (0, 0, 255) if (plateau_mask is not None and plateau_mask[j]) else color
            cv2.line(panel, p1, p2, draw_color, 2, cv2.LINE_AA)

    draw_text_with_bg(panel, f"{v_max}", (5, pad_t + 5), 0.45)
    draw_text_with_bg(panel, f"{v_min}", (5, pad_t + ch), 0.45)
    draw_text_with_bg(panel, f"{title}: {data[current_idx]:.1f}{unit}", (15, 30), 0.6, color, 2)

def draw_dist_panel(panel, dist, title, bin_range, color):
    ph, pw, _ = panel.shape
    pad_l, pad_r, pad_t, pad_b = 65, 30, 30, 40
    cw, ch = pw - pad_l - pad_r, ph - pad_t - pad_b
    cv2.rectangle(panel, (2, 2), (pw-2, ph-2), (18, 18, 18), -1)
    
    num_bins = len(dist)
    bw = cw / num_bins
    max_p = max(np.max(dist), 0.05)

    for b in range(num_bins):
        bh = int((dist[b] / max_p) * ch)
        px = int(pad_l + b * bw)
        cv2.rectangle(panel, (px, pad_t + ch - bh), (px + max(1, int(bw)), pad_t + ch), color, -1)

    draw_text_with_bg(panel, f"{bin_range[0]}", (pad_l, pad_t + ch + 20), 0.4)
    draw_text_with_bg(panel, f"{bin_range[1]}", (pad_l + cw - 40, pad_t + ch + 20), 0.4)
    draw_text_with_bg(panel, title, (15, 20), 0.5, (160, 160, 160))

def draw_quadrant(canvas_slice, trend_data, dist_data, current_idx, title, unit, color, y_range, plateau_mask=None):
    qh, qw, _ = canvas_slice.shape
    split_y = int(qh * 0.65)
    draw_trend_panel(canvas_slice[0:split_y, 0:qw], trend_data, current_idx, title, unit, color, y_range, plateau_mask)
    draw_dist_panel(canvas_slice[split_y:qh, 0:qw], dist_data, f"{title} Dist", y_range, tuple(int(c*0.8) for c in color))



def draw_predicted_remain_time_panel(panel, data, current_idx, title, unit, color, y_range):
    """Draw predicted remain time plot similar to plot_predicted_remain_time.py style"""
    ph, pw, _ = panel.shape
    pad_l, pad_r, pad_t, pad_b = 80, 40, 50, 50
    cw, ch = pw - pad_l - pad_r, ph - pad_t - pad_b

    # Background
    cv2.rectangle(panel, (2, 2), (pw-2, ph-2), (22, 22, 22), -1)

    v_min, v_max = y_range
    v_span = max(v_max - v_min, 1e-5)

    def to_y(val): return int(pad_t + ch - ((val - v_min) / v_span) * ch)
    def to_x(idx): return int(pad_l + (idx / (len(data)-1)) * cw)

    # Draw grid lines
    for i in range(5):
        y_val = v_min + i * (v_span / 4)
        y_pos = to_y(y_val)
        cv2.line(panel, (pad_l, y_pos), (pad_l + cw, y_pos), (40, 40, 40), 1)
        # Y-axis labels
        draw_text_with_bg(panel, f"{y_val:.1f}", (5, y_pos + 5), 0.4, (150, 150, 150))

    # Draw the data line
    if len(data) > 1:
        for j in range(1, current_idx + 1):
            p1 = (to_x(j-1), to_y(data[j-1]))
            p2 = (to_x(j), to_y(data[j]))
            cv2.line(panel, p1, p2, color, 2, cv2.LINE_AA)

    # Draw current position marker
    if current_idx < len(data):
        curr_x = to_x(current_idx)
        curr_y = to_y(data[current_idx])
        cv2.circle(panel, (curr_x, curr_y), 5, (255, 255, 255), -1)
        cv2.circle(panel, (curr_x, curr_y), 7, color, 2)

    # Title and current value
    draw_text_with_bg(panel, f"{title}: {data[current_idx]:.1f}{unit}", (15, 30), 0.6, color, 2)
    draw_text_with_bg(panel, f"Frame: {current_idx}/{len(data)-1}", (pw - 150, 30), 0.5, (180, 180, 180))

    # X-axis label
    draw_text_with_bg(panel, "Time (frames)", (pad_l + cw // 2 - 50, ph - 10), 0.45, (150, 150, 150))


def save_overall_remain_time_plot(p_times, save_dir, idx):
    """Save the overall predicted remain time plot as an image"""
    if not p_times:
        return

    # Create a large canvas for the overall plot
    img_w, img_h = 1920, 400
    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    global_max_t = max(max(p_times) if p_times else 1.0, 1.0)
    global_min_t = min(min(p_times) if p_times else 0.0, 0.0)

    # Draw the full plot with all data points
    draw_predicted_remain_time_panel(
        canvas,
        p_times,
        len(p_times) - 1,  # Show the full trajectory
        "Overall Predicted Remain Time",
        " frames",
        (0, 255, 255),  # Cyan color
        (global_min_t, global_max_t)
    )

    # Save the image
    img_file = os.path.join(save_dir, f"predict_remain_time_traj_{idx}.png")
    cv2.imwrite(img_file, canvas)
    print(f"Saved overall plot: {img_file}")


def linear_interpolate_times(key_times, data_len):
    """Linear interpolation: linearly interpolate time values between key frames"""
    key_indices = sorted(key_times.keys())
    key_values = [key_times[idx] for idx in key_indices]

    if len(key_indices) < 2:
        return np.full(data_len, key_values[0] if key_values else 0).tolist()

    # Use CubicSpline in linear mode
    cs = CubicSpline(key_indices, key_values, bc_type='natural')
    # To get linear interpolation, handle it manually or use np.interp
    all_indices = np.arange(data_len)
    p_times = np.interp(all_indices, key_indices, key_values)

    return p_times.tolist()


def cubic_spline_interpolate_times(key_times, data_len, bc_type='natural'):
    """
    Cubic spline interpolation: smooth interpolation using Cubic Spline

    Args:
        key_times: {frame_idx: predicted_time} dict of key frames
        data_len: total number of frames
        bc_type: boundary condition type
            - 'natural': natural boundary (second derivative = 0)
            - 'clamped': clamped boundary (first derivative = 0)
            - 'not-a-knot': not-a-knot boundary
            - 'periodic': periodic boundary
    """
    key_indices = sorted(key_times.keys())
    key_values = [key_times[idx] for idx in key_indices]

    if len(key_indices) < 2:
        return np.full(data_len, key_values[0] if key_values else 0).tolist()

    if len(key_indices) < 4:
        # Use linear interpolation when there are too few key points
        return linear_interpolate_times(key_times, data_len)

    # Create the cubic spline interpolator
    cs = CubicSpline(key_indices, key_values, bc_type=bc_type)
    all_indices = np.arange(data_len)
    p_times = cs(all_indices)

    return p_times.tolist()


def savitzky_golay_smooth(times, window_length=21, polyorder=3):
    """
    Savitzky-Golay filtering: smooth the time series data

    Args:
        times: time series data
        window_length: window length (must be odd and greater than polyorder)
        polyorder: polynomial order

    Returns:
        smoothed time series
    """
    times = np.array(times)

    # Ensure window_length is odd and does not exceed the data length
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, len(times))
    if window_length <= polyorder:
        polyorder = window_length - 2
    if window_length < 3:
        window_length = 3

    # Boundary handling: replicate boundary values to avoid edge effects from filtering
    pad_width = window_length // 2
    times_padded = np.pad(times, pad_width, mode='edge')

    # Apply Savitzky-Golay filtering
    smoothed = savgol_filter(times_padded, window_length, polyorder)

    # Remove padding
    return smoothed[pad_width:pad_width + len(times)].tolist()


def interpolate_times(key_times, data_len, method='linear', smooth=False,
                      smooth_window=21, smooth_polyorder=3, bc_type='natural'):
    """
    Unified interpolation interface supporting multiple interpolation and smoothing methods

    Args:
        key_times: {frame_idx: predicted_time} dict of key frames
        data_len: total number of frames
        method: interpolation method
            - 'linear': linear interpolation
            - 'cubic': cubic spline interpolation
        smooth: whether to apply Savitzky-Golay smoothing
        smooth_window: SG filter window length
        smooth_polyorder: SG filter polynomial order
        bc_type: spline interpolation boundary condition

    Returns:
        list of interpolated time series
    """
    # 1. Choose interpolation method
    if method == 'cubic':
        p_times = cubic_spline_interpolate_times(key_times, data_len, bc_type=bc_type)
    else:  # default: linear
        p_times = linear_interpolate_times(key_times, data_len)

    # 2. Optional: apply Savitzky-Golay smoothing
    if smooth:
        p_times = savitzky_golay_smooth(p_times, smooth_window, smooth_polyorder)

    return p_times


def load_task_map(data_path):
    """Build task_index -> task_text mapping from meta/tasks.jsonl"""
    import json
    from pathlib import Path
    # data_path: .../lerobot/<dataset>/data/chunk-XXX/episode_XXXXXX.parquet
    # tasks.jsonl: .../lerobot/<dataset>/meta/tasks.jsonl
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


def process_and_visualize(data_path, v_paths, agent, v_hl, t_hl, idx, save_dir,
                          history_offset=5,
                          stride=5, interpolation_method='linear', smooth=False,
                          smooth_window=21, smooth_polyorder=3, bc_type='natural'):
    # 1. Load data and video frames
    data = pd.read_parquet(data_path)
    task_map = load_task_map(data_path)
    frames_base = read_video_frames_pyav(v_paths['base'])
    frames_left = read_video_frames_pyav(v_paths['left'])
    frames_right = read_video_frames_pyav(v_paths['right'])

    if not frames_base:
        print(f"Warning: No frames found for {v_paths['base']}")
        return

    data_len = min(len(data), len(frames_base))

    # History frame offset (passed in via function argument)

    # --- Stage A: Inference stage (Inference - only run inference on key frames) ---
    print(f"--> [Trajectory {idx}] Predicting key frames (stride={stride})...")

    # Store inference results for key frames
    key_times = {}  # {frame_idx: predicted_time}

    key_indices = list(range(0, data_len, stride))
    # Ensure the last frame is also included
    if key_indices[-1] != data_len - 1:
        key_indices.append(data_len - 1)

    for i in tqdm(key_indices, desc="Inference"):
        # left_gripper = data['observation.state.puppet_left_gripper_position'][i]
        # right_gripper = data['observation.state.puppet_right_gripper_position'][i]
        # left_arm = data['observation.state.puppet_left_arm_position'][i]
        # right_arm = data['observation.state.puppet_right_arm_position'][i]
        # left_action_gripper = data['action.left_gripper_position'][i]
        # right_action_gripper = data['action.right_gripper_position'][i]
        # left_action_arm = data['action.left_arm_position'][i]
        # right_action_arm = data['action.right_arm_position'][i]
        
        # print("left_gripper shape: ", left_gripper.shape)
        # print("right_gripper shape: ", right_gripper.shape)
        # print("left_arm shape: ", left_arm.shape)
        # print("right_arm shape: ", right_arm.shape)
        # print("left_action_gripper shape: ", left_action_gripper.shape)
        # print("right_action_gripper shape: ", right_action_gripper.shape)
        # print("left_action_arm shape: ", left_action_arm.shape)
        # print("right_action_arm shape: ", right_action_arm.shape)
        
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

        # Get history frame index
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
        print("the prompt is: ", batch['prompt'])

        with torch.no_grad():
            preds = agent.predict_value(batch)
            t_soft = preds['remain_time'].cpu()

            # Decode scalar value for the trend plot (Trend)
            key_times[i] = t_hl.batch_decode(t_soft).item()

    # --- Stage B: Interpolate to generate predicted values for all frames ---
    smooth_str = " + Savitzky-Golay smoothing" if smooth else ""
    print(f"--> [Trajectory {idx}] Interpolating {len(key_times)} key frames to {data_len} frames ({interpolation_method}{smooth_str})...")
    p_times = interpolate_times(key_times, data_len,
                                method=interpolation_method,
                                smooth=smooth,
                                smooth_window=smooth_window,
                                smooth_polyorder=smooth_polyorder,
                                bc_type=bc_type)

    # --- Stage B: Compute the global max (Global Max for Y-axis) ---
    global_max_t = max(max(p_times) if p_times else 1.0, 1.0)
    global_min_t = min(min(p_times) if p_times else 0.0, 0.0)

    # --- Stage C: Rendering stage (Rendering) ---
    orig_h, orig_w, _ = frames_base[0].shape

    # Calculate dimensions: top row has 3 videos, bottom row has 1 wide plot
    # Video panel dimensions (keep original aspect ratio)
    video_panel_w = orig_w
    video_panel_h = orig_h

    # Top row: 3 video panels side by side
    top_row_width = video_panel_w * 3
    top_row_height = video_panel_h

    # Bottom row: predicted_remain_time plot with 5:1 aspect ratio width relative to height
    # We'll make it span the full width of the top row
    plot_height = int(top_row_width / 5)  # 5:1 aspect ratio means width:height = 5:1

    # Total canvas dimensions
    canvas_w = top_row_width
    canvas_h = top_row_height + plot_height

    out_file = os.path.join(save_dir, f"viz_3x1_traj_{idx}.mp4")
    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (canvas_w, canvas_h))

    for i in tqdm(range(data_len), desc="Rendering"):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # === Top Row: Three Camera Views ===
        # Base Camera (Left)
        frame_base_pp = preprocess_frame_for_display(frames_base[i])
        v_f_base = cv2.cvtColor(frame_base_pp, cv2.COLOR_RGB2BGR)
        canvas[0:video_panel_h, 0:video_panel_w] = cv2.resize(v_f_base, (video_panel_w, video_panel_h))
        draw_text_with_bg(canvas, "BASE CAMERA", (20, 40), 0.8, (255, 255, 255), 2)

        # Left Camera (Center)
        frame_left_pp = preprocess_frame_for_display(frames_left[i])
        v_f_left = cv2.cvtColor(frame_left_pp, cv2.COLOR_RGB2BGR)
        canvas[0:video_panel_h, video_panel_w:video_panel_w*2] = cv2.resize(v_f_left, (video_panel_w, video_panel_h))
        draw_text_with_bg(canvas, "LEFT CAMERA", (video_panel_w + 20, 40), 0.8, (255, 255, 255), 2)

        # Right Camera (Right)
        frame_right_pp = preprocess_frame_for_display(frames_right[i])
        v_f_right = cv2.cvtColor(frame_right_pp, cv2.COLOR_RGB2BGR)
        canvas[0:video_panel_h, video_panel_w*2:video_panel_w*3] = cv2.resize(v_f_right, (video_panel_w, video_panel_h))
        draw_text_with_bg(canvas, "RIGHT CAMERA", (video_panel_w*2 + 20, 40), 0.8, (255, 255, 255), 2)

        # === Bottom Row: Predicted Remain Time Plot ===
        plot_y_start = top_row_height
        plot_y_end = canvas_h

        # Create a slice for the plot panel
        plot_panel = canvas[plot_y_start:plot_y_end, 0:canvas_w]

        # Draw the predicted remain time plot
        draw_predicted_remain_time_panel(
            plot_panel,
            p_times,
            i,
            "Predicted Remain Time",
            " frames",
            (0, 255, 255),  # Cyan color
            (global_min_t, global_max_t)
        )

        out.write(canvas)

    out.release()
    print(f"Finished: {out_file}")

    # --- Stage D: Save the overall predict_remain_time plot ---
    save_overall_remain_time_plot(p_times, save_dir, idx)

# --- 4. Configuration parameter class ---

class InterpolationConfig:
    """Interpolation and smoothing configuration"""
    def __init__(self,
                 stride: int = 10,
                 interpolation_method: str = 'cubic',  # 'linear' or 'cubic'
                 smooth: bool = True,
                 smooth_window: int = 21,
                 smooth_polyorder: int = 3,
                 bc_type: str = 'natural'):  # 'natural', 'clamped', 'not-a-knot', 'periodic'
        self.stride = stride
        self.interpolation_method = interpolation_method
        self.smooth = smooth
        self.smooth_window = smooth_window
        self.smooth_polyorder = smooth_polyorder
        self.bc_type = bc_type

    def __str__(self):
        smooth_str = f" + SG(window={self.smooth_window}, poly={self.smooth_polyorder})" if self.smooth else ""
        return f"{self.interpolation_method}_interpolation{smooth_str}"


# --- 5. Main entry point ---

def main(interp_config: InterpolationConfig = None):
    """
    Args:
        interp_config: interpolation configuration; uses the default configuration if None

    Usage Examples:
        # Default configuration (linear interpolation)
        main()

        # Cubic spline interpolation
        config = InterpolationConfig(interpolation_method='cubic')
        main(config)

        # Linear interpolation + Savitzky-Golay smoothing
        config = InterpolationConfig(smooth=True, smooth_window=21, smooth_polyorder=3)
        main(config)

        # Cubic spline + smoothing
        config = InterpolationConfig(interpolation_method='cubic', smooth=True)
        main(config)
    """
    if interp_config is None:
        interp_config = InterpolationConfig()  # default configuration

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

    # data_dir = "/mnt/dataset/toago6/block_x_humanoid_data/lerobot_version/x_humanoid_lerobot_data_with_quality_41_01_14"
    # data_dir = "/mnt/dataset/toago6/workspace/code/rl_block_x_humanoid_data/finished_annotation_pretraining_manipulation_data/x_humanoid_lerobot_data_with_quality_16_01_21_processed_20260129_processed_20260202"
    # save_dir = "value_debug/seen_other_data"
    save_dir = config.vis_dir
    data_dir = config.vis_data_dir
    os.makedirs(save_dir, exist_ok=True)

    parquet_dir = os.path.join(data_dir, "data", "chunk-000")
    files = sorted([f for f in os.listdir(parquet_dir) if f.endswith(".parquet")])
    print(len(files))
    for i, name in enumerate(files):

        p_path = os.path.join(parquet_dir, name)
        v_paths = {
            'base': os.path.join(data_dir, "videos/chunk-000/observation.image.image", name.replace(".parquet", ".mp4")),
            'left': os.path.join(data_dir, "videos/chunk-000/observation.image.left", name.replace(".parquet", ".mp4")),
            'right': os.path.join(data_dir, "videos/chunk-000/observation.image.right", name.replace(".parquet", ".mp4"))
        }
        process_and_visualize(p_path, v_paths, agent, v_hl, t_hl, i, save_dir,
                             history_offset=config.history_length,
                             stride=interp_config.stride,
                             interpolation_method=interp_config.interpolation_method,
                             smooth=interp_config.smooth,
                             smooth_window=interp_config.smooth_window,
                             smooth_polyorder=interp_config.smooth_polyorder,
                             bc_type=interp_config.bc_type)

if __name__ == "__main__":
    main()