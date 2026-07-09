"""
Multi-dimensional quality assessment visualization - 3x3 Matrix Layout.

Layout:
┌─────────┬─────────┬─────────┐
│ Main[t] │ Left[t] │ Right[t]│  <- Current frame (t)
├─────────┼─────────┼─────────┤
│Main[t+50]│Left[t+50]│Right[t+50]│ <- 50 frames later
├─────────┴─────────┴─────────┤
│     Remain Time Curve (Quality)    │
└───────────────────────────────────┘
"""

import os
import cv2
import numpy as np
import av
import pandas as pd
from typing import Tuple, Dict
def get_quality_color_bgr(quality: int) -> Tuple[int, int, int]:
    """Get BGR color for quality label.

    Quality mapping (chunk-level):
    - quality = 2: GOOD (Green)
    - quality = 1: MEDIUM (Yellow)
    - quality = 0: BAD (Red)
    """
    if quality == 2:
        return (60, 220, 60)   # Green - Good (POSITIVE)
    elif quality == 1:
        return (60, 200, 230)  # Yellow - Medium
    else:  # quality == 0
        return (50, 50, 230)   # Red - Bad (NEGATIVE)


def get_quality_name(quality: int) -> str:
    """Get quality label name.

    Quality mapping (chunk-level):
    - quality = 2: GOOD
    - quality = 1: MEDIUM
    - quality = 0: BAD
    """
    names = {2: "GOOD", 1: "MEDIUM", 0: "BAD"}
    return names.get(quality, "UNKNOWN")


def resize_and_pad(img: np.ndarray, target_size: int = 256,
                   pad_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Resize image to square and pad if needed."""
    h, w = img.shape[:2]

    # Resize to fit within target_size while maintaining aspect ratio
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to square
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=pad_color)

    return padded


def add_quality_overlay(img: np.ndarray, quality: int, alpha: float = 0.3):
    """Add semi-transparent quality color overlay."""
    h, w = img.shape[:2]
    color = get_quality_color_bgr(quality)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Add colored border
    border_width = 8
    cv2.rectangle(img, (0, 0), (w, h), color, border_width)

    # Add text label
    label = get_quality_name(quality)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    text_x = (w - text_w) // 2
    text_y = 40

    # Text background
    cv2.rectangle(img, (text_x - 10, text_y - text_h - 10),
                 (text_x + text_w + 10, text_y + 10), (0, 0, 0), -1)

    cv2.putText(img, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

    return img


def add_frame_label(img: np.ndarray, label: str, position: str = "top-left"):
    """Add frame label (e.g., "t", "t+50", camera names)."""
    font_scale = 0.9
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    if position == "top-left":
        x, y = 15, 35
        cv2.rectangle(img, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    elif position == "top-center":
        x, y = (img.shape[1] - text_w) // 2, 35
        cv2.rectangle(img, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_remain_time_curve(
    canvas: np.ndarray,
    remain_time: np.ndarray,
    quality: np.ndarray,
    current_idx: int,
    chunk_size: int,
    y_range: Tuple[float, float]
):
    """
    Draw remain_time curve with quality-based coloring and chunk markers.

    Args:
        canvas: Canvas to draw on
        remain_time: Array of remain_time values
        quality: Array of quality labels
        current_idx: Current frame index
        chunk_size: Size of each chunk (50)
        y_range: (y_min, y_max) for scaling
    """
    h, w = canvas.shape[:2]
    pad_l, pad_r, pad_t, pad_b = 80, 40, 60, 60
    cw, ch = w - pad_l - pad_r, h - pad_t - pad_b

    # Background
    cv2.rectangle(canvas, (0, 0), (w, h), (18, 18, 18), -1)

    y_min, y_max = y_range
    y_span = max(y_max - y_min, 1e-5)

    def to_y(val):
        return int(pad_t + ch - ((val - y_min) / y_span) * ch)

    def to_x(idx):
        if len(remain_time) <= 1:
            return pad_l
        return int(pad_l + (idx / (len(remain_time) - 1)) * cw)

    # Draw grid lines
    for i in range(5):
        y_val = y_min + i * (y_span / 4)
        y = to_y(y_val)
        cv2.line(canvas, (pad_l, y), (pad_l + cw, y), (40, 40, 40), 1)
        cv2.putText(canvas, f"{y_val:.0f}", (10, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Draw remain_time curve with quality coloring
    if len(remain_time) > 1:
        for j in range(1, min(current_idx + 1, len(remain_time))):
            p1 = (to_x(j - 1), to_y(remain_time[j - 1]))
            p2 = (to_x(j), to_y(remain_time[j]))

            # Color based on quality
            q = quality[j] if j < len(quality) else 1
            color = get_quality_color_bgr(q)

            # Line width: thicker for GOOD/BAD, thinner for MEDIUM
            thickness = 3 if q != 1 else 2
            cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)

    # Draw chunk boundaries (vertical lines)
    for chunk_start in range(0, len(remain_time), chunk_size):
        x = to_x(chunk_start)
        cv2.line(canvas, (x, pad_t), (x, pad_t + ch), (100, 100, 100), 1, cv2.LINE_AA)

    # Highlight current chunk
    current_chunk_start = (current_idx // chunk_size) * chunk_size
    x_start = to_x(current_chunk_start)
    x_end = to_x(min(current_chunk_start + chunk_size, len(remain_time)))

    # Semi-transparent overlay for current chunk
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x_start, pad_t), (x_end, pad_t + ch),
                  (80, 80, 80), -1)
    cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

    # Draw current chunk label
    current_quality = quality[current_idx] if current_idx < len(quality) else 1
    chunk_label = get_quality_name(current_quality)
    chunk_color = get_quality_color_bgr(current_quality)

    # Text at top
    cv2.putText(canvas, f"Chunk Quality: {chunk_label}",
                (pad_l, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, chunk_color, 2, cv2.LINE_AA)

    # Current position cursor
    cursor_x = to_x(current_idx)
    cursor_y = to_y(remain_time[current_idx])

    # Cursor line
    cv2.line(canvas, (cursor_x, pad_t), (cursor_x, pad_t + ch), (200, 200, 200), 2, cv2.LINE_AA)

    # Cursor point
    cv2.circle(canvas, (cursor_x, cursor_y), 10, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cursor_x, cursor_y), 12, (0, 0, 0), 2, cv2.LINE_AA)

    # Y-axis label
    cv2.putText(canvas, "Remain Time", (10, pad_t - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    # Current value
    cv2.putText(canvas, f"{remain_time[current_idx]:.1f}",
                (cursor_x + 10, cursor_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # X-axis label (frame)
    cv2.putText(canvas, f"Frame: {current_idx}",
                (w - 150, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    # Legend
    legend_y = pad_t + 10
    legend_x = w - 200
    for q, name in [(2, "GOOD"), (1, "MEDIUM"), (0, "BAD")]:
        color = get_quality_color_bgr(q)
        cv2.rectangle(canvas, (legend_x, legend_y), (legend_x + 20, legend_y + 15), color, -1)
        cv2.putText(canvas, name, (legend_x + 28, legend_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        legend_y += 22


def create_multidim_visualization_v2(
    df: pd.DataFrame,
    video_paths: Dict[str, str],
    output_path: str,
    img_size: int = 256,
    chunk_size: int = 50
):
    """
    Create 3x3 matrix visualization.

    Layout:
    ┌─────────┬─────────┬─────────┐
    │ Main[t] │ Left[t] │ Right[t]│  <- Current frame
    ├─────────┼─────────┼─────────┤
    │Main[t+50]│Left[t+50]│Right[t+50]│ <- 50 frames later
    ├─────────┴─────────┴─────────┤
    │     Remain Time Curve         │
    └──────────────────────────────┘

    Args:
        df: DataFrame with quality columns
        video_paths: Dictionary with camera paths
        output_path: Path to output video
        img_size: Size of each camera view (default 256)
        chunk_size: Size of each chunk (default 50)
    """
    # Check if required videos exist
    required_videos = ['main', 'left', 'right']
    for vid in required_videos:
        if vid not in video_paths or not os.path.exists(video_paths[vid]):
            print(f"  Warning: Video '{vid}' not found: {video_paths.get(vid, 'N/A')}")
            return

    # Get data from DataFrame
    quality = df['quality'].values
    remain_time = df.get('predict_remain_time', pd.Series([0]*len(df))).values

    # Open video containers
    containers = {}
    streams = {}

    for vid_name in required_videos:
        try:
            containers[vid_name] = av.open(video_paths[vid_name])
            streams[vid_name] = containers[vid_name].streams.video[0]
        except Exception as e:
            print(f"  Error opening {vid_name} video: {e}")
            return

    # Get FPS from main video
    fps = float(streams['main'].average_rate) if streams['main'].average_rate else 30.0

    # Determine video length
    T_data = len(df)
    T_min = min([
        streams['main'].frames if streams['main'].frames else T_data,
        streams['left'].frames if streams['left'].frames else T_data,
        streams['right'].frames if streams['right'].frames else T_data,
        T_data
    ])

    if T_min == 0:
        print(f"  Warning: Empty data or videos")
        for c in containers.values():
            c.close()
        return

    T = T_min

    # Compute y_range for remain_time
    rt_valid = remain_time[remain_time < 3000]
    if len(rt_valid) > 0:
        y_min, y_max = float(np.min(rt_valid)), float(np.max(rt_valid))
        y_pad = 0.1 * (y_max - y_min)
        y_min -= y_pad
        y_max += y_pad
    else:
        y_min, y_max = 0, 100

    # Calculate output dimensions
    # 3x3 grid: each cell is img_size x img_size
    # Top row: 3 views (current)
    # Middle row: 3 views (t+50)
    # Bottom row: 1 curve (spans full width)
    grid_cols = 3
    grid_rows = 3

    output_w = img_size * grid_cols
    output_h = img_size * grid_rows

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None

    print(f"  Creating 3x3 visualization: {T} frames")

    # First pass: cache all frames
    print(f"  Caching frames...")
    frame_cache = {name: [] for name in required_videos}

    decoders = {
        name: containers[name].decode(streams[name])
        for name in required_videos
    }

    for i in range(T):
        for name in required_videos:
            try:
                frame = next(decoders[name])
                img_rgb = frame.to_ndarray(format='rgb24')
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                frame_cache[name].append(img_bgr)
            except StopIteration:
                # If one video ends, pad with black frames
                if len(frame_cache[name]) < i + 1:
                    while len(frame_cache[name]) < T:
                        frame_cache[name].append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
                break

    # Ensure all caches have T frames
    for name in required_videos:
        while len(frame_cache[name]) < T:
            frame_cache[name].append(np.zeros((img_size, img_size, 3), dtype=np.uint8))

    print(f"  Generating visualization...")

    for i in range(T):
        # Get current frame (t) from cache
        frames_t = {}
        for name in required_videos:
            frames_t[name] = frame_cache[name][i]

        # Get frame at t+50 from cache
        frames_t50 = {}
        t50_idx = min(i + 50, T - 1)
        for name in required_videos:
            frames_t50[name] = frame_cache[name][t50_idx]

        # Resize and pad all frames to img_size
        for name in required_videos:
            frames_t[name] = resize_and_pad(frames_t[name], img_size)
            frames_t50[name] = resize_and_pad(frames_t50[name], img_size)

        # Get current quality
        q = quality[i] if i < len(quality) else 1

        # Add quality overlay to current frames
        for name in required_videos:
            frames_t[name] = add_quality_overlay(frames_t[name].copy(), q, alpha=0.15)

        # Add labels
        camera_labels = {'main': 'Main Camera', 'left': 'Left Wrist', 'right': 'Right Wrist'}

        # Top row labels
        for idx, name in enumerate(['main', 'left', 'right']):
            add_frame_label(frames_t[name], camera_labels[name], "top-center")
            add_frame_label(frames_t[name], "t", "top-left")

        # Middle row labels
        for idx, name in enumerate(['main', 'left', 'right']):
            add_frame_label(frames_t50[name], "t+50", "top-left")

        # Create output frame (3x3 grid)
        output_frame = np.zeros((output_h, output_w, 3), dtype=np.uint8)

        # Top row: current frames
        output_frame[0:img_size, 0:img_size] = frames_t['main']
        output_frame[0:img_size, img_size:img_size*2] = frames_t['left']
        output_frame[0:img_size, img_size*2:img_size*3] = frames_t['right']

        # Middle row: t+50 frames
        output_frame[img_size:img_size*2, 0:img_size] = frames_t50['main']
        output_frame[img_size:img_size*2, img_size:img_size*2] = frames_t50['left']
        output_frame[img_size:img_size*2, img_size*2:img_size*3] = frames_t50['right']

        # Bottom row: remain_time curve (spans full width)
        curve_canvas = output_frame[img_size*2:img_size*3, :]
        draw_remain_time_curve(
            curve_canvas,
            remain_time,
            quality,
            i,
            chunk_size,
            (y_min, y_max)
        )

        # Add grid lines for visual separation
        # Horizontal lines
        cv2.line(output_frame, (0, img_size), (output_w, img_size), (80, 80, 80), 2)
        cv2.line(output_frame, (0, img_size*2), (output_w, img_size*2), (80, 80, 80), 2)

        # Vertical lines
        cv2.line(output_frame, (img_size, 0), (img_size, img_size*2), (80, 80, 80), 2)
        cv2.line(output_frame, (img_size*2, 0), (img_size*2, img_size*2), (80, 80, 80), 2)

        # Initialize writer
        if writer is None:
            writer = cv2.VideoWriter(output_path, fourcc, fps, (output_w, output_h))

        writer.write(output_frame)

    # Cleanup
    if writer is not None:
        writer.release()

    for c in containers.values():
        c.close()

    print(f"  Saved: {output_path}")
