"""
Generate videos with smoothed predict_remain_time and rank-based quality with minimum score threshold.

This script:
1. Computes quality scores based on predict_remain_time
2. Overwrites the 'quality' field in original parquet files with new quality values
3. Generates visualization videos with the computed quality

Quality assessment based on global ranking with MINIMUM SCORE THRESHOLD:
- For each 50-frame chunk: sum(prt[t] - prt[t+50]) for all t in chunk
- Rank all chunks across all episodes
- Top 30% chunks: GOOD action (green), BUT if score < 20 -> MEDIUM
- Middle 40% chunks (30-70%): MEDIUM action (cyan)
- Bottom 30% chunks: BAD action (red)

Video layout (3x3 grid):
┌─────────┬─────────┬─────────┐
│ Main[t] │ Left[t] │ Right[t]│  <- Current frame (t)
├─────────┼─────────┼─────────┤
│Main[t+50]│Left[t+50]│Right[t+50]│ <- 50 frames later
├─────────┴─────────┴─────────┤
│     Remain Time Curve         │
└──────────────────────────────┘
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import cv2
import av
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from multi_dim_visualization_v2 import (
    resize_and_pad,
    add_quality_overlay,
    add_frame_label,
    draw_remain_time_curve
)


# Paths

# Default data directory (can be overridden via --data_dir command-line argument)
DEFAULT_DATA_DIR = Path("/mnt/dataset/toago6/workspace/code/rl_block_x_humanoid_data/stack_pretraining_all_data_soft_version")


# Camera names in the video directory
CAMERA_MAPPING = {
    'main': 'observation.image.image',
    'left': 'observation.image.left',
    'right': 'observation.image.right'
}
CHUNK_SIZE = 60
# Quality thresholds
QUALITY_TOP_PERCENTILE = 0.5    # Top 30% = GOOD
QUALITY_BOTTOM_PERCENTILE = 0.3  # Bottom 30% = BAD (middle 40% = MEDIUM)
MIN_GOOD_SCORE = -10               # Minimum score required for GOOD (otherwise -> MEDIUM)

# Intervention parameters
INTERVENTION_PRE_ONSET_FRAMES = 30       # frames before intervention onset to mark as BAD
INTERVENTION_ONSET_SKIP_FRAMES = 20      # first N frames of each intervention use value function estimate (no override)

# Last N frames forced to GOOD
LAST_GOOD_FRAMES = 0

# Minimum consecutive frames required to maintain quality level
# If a run of identical quality is shorter than this, it gets downgraded (GOOD->MEDIUM, MEDIUM->BAD)
MIN_CONSECUTIVE_FRAMES = 30


def compute_chunk_scores(prt: np.ndarray, chunk_size: int = 50) -> np.ndarray:
    """
    Compute scores for each position in the trajectory.

    Score[t] = sum of (prt[i] - prt[i+50]) for i in the chunk containing t

    Args:
        prt: Array of predicted_remain_time values
        chunk_size: Size of each chunk (default 50)

    Returns:
        Array of scores, same length as prt
    """
    n = len(prt)
    scores = np.zeros(n)

    # Compute diff for each position: prt[t] - prt[t+50]
    diffs = np.zeros(n)
    for t in range(n):
        t_plus_50 = min(t + 50, n - 1)
        diffs[t] = prt[t] - prt[t_plus_50] - (t_plus_50 - t)
        # diffs[t] = n - t - prt[t]
        scores[t] = diffs[t]

    return scores


def compute_quality_from_global_ranking_with_min_score(
    all_scores: List[np.ndarray],
    min_good_score: float = MIN_GOOD_SCORE
) -> List[np.ndarray]:
    """
    Compute quality based on global ranking across all episodes,
    but GOOD actions must have score >= min_good_score.

    Args:
        all_scores: List of score arrays, one per episode
        min_good_score: Minimum score required for GOOD quality

    Returns:
        List of quality arrays (0=BAD, 1=MEDIUM, 2=GOOD), one per episode
    """
    # Collect all scores with their episode and position info
    score_entries = []
    for episode_idx, scores in enumerate(all_scores):
        for pos_idx, score in enumerate(scores):
            score_entries.append({
                'episode': episode_idx,
                'position': pos_idx,
                'score': score
            })

    # Sort by score (HIGHER is better - we want MAXIMUM remain_time reduction)
    score_entries.sort(key=lambda x: x['score'], reverse=True)

    # Assign quality based on percentiles
    n = len(score_entries)
    top_threshold_idx = int(n * QUALITY_TOP_PERCENTILE)
    bottom_threshold_idx = int(n * (1 - QUALITY_BOTTOM_PERCENTILE))

    # Higher scores = better (more reduction in remain_time)
    # So top 30% (highest scores) = GOOD (2) - but only if score >= min_good_score
    # Middle 40% = MEDIUM (1)
    # Bottom 30% (lowest scores) = BAD (0)

    quality_by_ep_pos = {}
    demoted_to_medium_count = 0
    medium_to_bad_count = 0

    for idx, entry in enumerate(score_entries):
        score = entry['score']

        if idx < top_threshold_idx:
            # Originally in top percentile -> would be GOOD
            # But check minimum score threshold
            if score >= min_good_score:
                quality = 2  # GOOD
            else:
                quality = 1  # Demoted to MEDIUM (score too low)
                demoted_to_medium_count += 1
        elif idx < bottom_threshold_idx:
            # MEDIUM, but demote to BAD if score < 0
            if score < 0:
                quality = 0  # BAD (negative score)
                medium_to_bad_count += 1
            else:
                quality = 1  # MEDIUM
        else:
            quality = 0  # BAD

        key = (entry['episode'], entry['position'])
        quality_by_ep_pos[key] = quality

    print(f"    Demoted {demoted_to_medium_count} actions from GOOD to MEDIUM (score < {min_good_score})")
    print(f"    Demoted {medium_to_bad_count} actions from MEDIUM to BAD (score < 0)")

    # Build quality arrays for each episode
    all_qualities = []
    for episode_idx, scores in enumerate(all_scores):
        n = len(scores)
        qualities = np.zeros(n, dtype=int)
        for pos_idx in range(n):
            key = (episode_idx, pos_idx)
            qualities[pos_idx] = quality_by_ep_pos.get(key, 1)
        all_qualities.append(qualities)

    return all_qualities


def apply_intervention_override(
    all_qualities: List[np.ndarray],
    global_all_data: List[dict],
    pre_onset_frames: int = INTERVENTION_PRE_ONSET_FRAMES,
    onset_skip_frames: int = INTERVENTION_ONSET_SKIP_FRAMES,
) -> List[np.ndarray]:
    """
    Override quality based on human_intervention, applied AFTER proportional/ranking processing.
    - pre_onset_frames frames before each intervention onset -> quality = 0 (BAD)
    - First onset_skip_frames of each intervention segment -> keep value function estimate (no override)
    - Remaining intervention frames (after onset_skip_frames) -> quality = 2 (GOOD)
    """
    override_counts = {'good': 0, 'bad': 0, 'vf_kept': 0}

    for idx, data in enumerate(global_all_data):
        df = data['df']
        quality = all_qualities[idx].copy()
        n = len(quality)

        if 'human_intervention' not in df.columns:
            continue

        hi_raw = df['human_intervention'].values
        if hi_raw.ndim > 1:
            hi_raw = hi_raw.squeeze()
        human_intervention = np.asarray(hi_raw).flatten()

        # Find intervention onset indices (transition from 0 to 1)
        onset_indices = []
        for i in range(1, min(n, len(human_intervention))):
            if human_intervention[i] == 1 and human_intervention[i - 1] == 0:
                onset_indices.append(i)

        # Mark pre-onset frames as BAD (quality=0)
        for onset in onset_indices:
            start = max(0, onset - pre_onset_frames)
            for i in range(start, onset):
                if human_intervention[i] == 0:  # only override non-intervention frames
                    quality[i] = 0
                    override_counts['bad'] += 1

        # For each intervention segment:
        #   - first onset_skip_frames: keep value function estimate (no override)
        #   - remaining frames: set to GOOD (human correction is correct)
        for onset in onset_indices:
            # Find end of this intervention segment
            seg_end = onset
            while seg_end < min(n, len(human_intervention)) and human_intervention[seg_end] == 1:
                seg_end += 1

            skip_end = min(onset + onset_skip_frames, seg_end)

            # First onset_skip_frames: no override (keep value function estimate)
            override_counts['vf_kept'] += skip_end - onset

            # Remaining intervention frames: GOOD (human correction is correct)
            for i in range(skip_end, seg_end):
                quality[i] = 2
                override_counts['good'] += 1

        all_qualities[idx] = quality

    total_overrides = override_counts['good'] + override_counts['bad']
    if total_overrides > 0 or override_counts['vf_kept'] > 0:
        print(f"    Intervention override: GOOD(after first {onset_skip_frames}f)={override_counts['good']}, "
              f"BAD(pre-onset {pre_onset_frames}f)={override_counts['bad']}, "
              f"VF-kept(first {onset_skip_frames}f)={override_counts['vf_kept']}")

    return all_qualities


def apply_consecutive_consistency(
    all_qualities: List[np.ndarray],
    min_consecutive: int = MIN_CONSECUTIVE_FRAMES
) -> List[np.ndarray]:
    """
    Enforce consecutive-frame consistency on quality labels.
    If a run of identical quality is shorter than min_consecutive frames,
    downgrade it: GOOD(2)->MEDIUM(1), MEDIUM(1)->BAD(0), BAD(0) stays BAD(0).

    Args:
        all_qualities: List of quality arrays, one per episode
        min_consecutive: Minimum number of consecutive frames required

    Returns:
        List of updated quality arrays
    """
    total_downgraded = {2: 0, 1: 0}  # count frames downgraded from GOOD and MEDIUM

    for idx in range(len(all_qualities)):
        quality = all_qualities[idx].copy()
        n = len(quality)
        if n == 0:
            continue

        # Find runs of consecutive identical values
        i = 0
        while i < n:
            current_val = quality[i]
            run_start = i
            while i < n and quality[i] == current_val:
                i += 1
            run_len = i - run_start

            # If run is too short and quality > 0, downgrade
            if run_len < min_consecutive and current_val > 0:
                downgraded_val = current_val - 1
                for j in range(run_start, run_start + run_len):
                    quality[j] = downgraded_val
                total_downgraded[current_val] += run_len

        all_qualities[idx] = quality

    print(f"    Consecutive consistency (min {min_consecutive} frames):")
    print(f"      GOOD->MEDIUM: {total_downgraded[2]} frames")
    print(f"      MEDIUM->BAD: {total_downgraded[1]} frames")

    return all_qualities


def get_video_paths(dir_path: str, episode_idx: int) -> Dict[str, str]:
    """Get paths to all three camera videos for an episode."""
    paths = {}
    for camera_name, dir_name in CAMERA_MAPPING.items():
        video_path = os.path.join(dir_path,"videos","chunk-000", dir_name, f"episode_{episode_idx:06d}.mp4")
        paths[camera_name] = str(video_path)
    return paths


def create_smoothed_visualization_video(
    df: pd.DataFrame,
    quality: np.ndarray,
    video_paths: Dict[str, str],
    output_path: str,
    prt_column: str = 'predict_remain_time_smoothed',
    img_size: int = 256,
    chunk_size: int = 50
):
    """
    Create 3x3 matrix visualization with smoothed predict_remain_time.

    Args:
        df: DataFrame with smoothed predict_remain_time column
        quality: Quality array (0=BAD, 1=MEDIUM, 2=GOOD)
        video_paths: Dictionary with camera paths
        output_path: Path to output video
        prt_column: Column name for predict_remain_time (original or smoothed)
        img_size: Size of each camera view (default 256)
        chunk_size: Size of each chunk (default 50)
    """
    # Check if required videos exist
    required_videos = ['main', 'left', 'right']
    for vid in required_videos:
        if vid not in video_paths or not os.path.exists(video_paths[vid]):
            print(f"    Warning: Video '{vid}' not found: {video_paths.get(vid, 'N/A')}")
            return False

    # Get smoothed data
    prt = df[prt_column].values

    # Open video containers
    containers = {}
    streams = {}

    for vid_name in required_videos:
        try:
            containers[vid_name] = av.open(video_paths[vid_name])
            streams[vid_name] = containers[vid_name].streams.video[0]
        except Exception as e:
            print(f"    Error opening {vid_name} video: {e}")
            return False

    # Get FPS from main video
    fps = float(streams['main'].average_rate) if streams['main'].average_rate else 30.0
    fps = int(fps)  # Convert to integer for compatibility

    # Determine video length
    T_data = len(df)
    T_min = min([
        streams['main'].frames if streams['main'].frames else T_data,
        streams['left'].frames if streams['left'].frames else T_data,
        streams['right'].frames if streams['right'].frames else T_data,
        T_data
    ])

    if T_min == 0:
        print(f"    Warning: Empty data or videos")
        for c in containers.values():
            c.close()
        return False

    T = T_min

    # Compute y_range for remain_time
    rt_valid = prt[prt < 5000]
    if len(rt_valid) > 0:
        y_min, y_max = float(np.min(rt_valid)), float(np.max(rt_valid))
        y_pad = 0.1 * (y_max - y_min)
        y_min -= y_pad
        y_max += y_pad
    else:
        y_min, y_max = 0, 100

    # Calculate output dimensions
    grid_cols = 3
    grid_rows = 3

    output_w = img_size * grid_cols
    output_h = img_size * grid_rows

    # Create output video writer using PyAV for better compatibility
    # Use H.264 codec which is compatible with QuickTime Player
    output_container = av.open(output_path, 'w')
    output_stream = output_container.add_stream('libx264', rate=fps)
    output_stream.width = output_w
    output_stream.height = output_h
    output_stream.pix_fmt = 'yuv420p'

    # Set codec options for better quality and compatibility
    output_stream.codec_context.options = {
        'crf': '23',  # Quality factor (lower = better quality)
        'preset': 'fast',  # Encoding speed
    }

    print(f"    Creating visualization: {T} frames")

    # First pass: cache all frames
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
                # Pad with black frames if video ends
                if len(frame_cache[name]) < i + 1:
                    while len(frame_cache[name]) < T:
                        frame_cache[name].append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
                break

    # Ensure all caches have T frames
    for name in required_videos:
        while len(frame_cache[name]) < T:
            frame_cache[name].append(np.zeros((img_size, img_size, 3), dtype=np.uint8))

    print(f"    Generating visualization...")

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

        # Get current quality (from pre-computed array)
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
            prt,
            quality,
            i,
            chunk_size,
            (y_min, y_max)
        )

        # Add grid lines for visual separation
        cv2.line(output_frame, (0, img_size), (output_w, img_size), (80, 80, 80), 2)
        cv2.line(output_frame, (0, img_size*2), (output_w, img_size*2), (80, 80, 80), 2)
        cv2.line(output_frame, (img_size, 0), (img_size, img_size*2), (80, 80, 80), 2)
        cv2.line(output_frame, (img_size*2, 0), (img_size*2, img_size*2), (80, 80, 80), 2)

        # Convert BGR to RGB for PyAV
        output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

        # Create video frame and encode
        video_frame = av.VideoFrame.from_ndarray(output_frame_rgb, format='rgb24')
        for packet in output_stream.encode(video_frame):
            output_container.mux(packet)

    # Flush encoder
    for packet in output_stream.encode():
        output_container.mux(packet)

    # Close output container
    output_container.close()

    for c in containers.values():
        c.close()

    print(f"    Saved: {output_path}")
    return True


def process_videos_for_method(method_name, data_dir, start_idx: int = 0, end_idx: int = None):
    """
    Generate videos for a specific smoothing method with GLOBAL rank-based quality and minimum score threshold.

    Quality is computed ACROSS ALL FOLDERS (global ranking), not per-folder.

    Args:
        method_name: One of 'savgol', 'ema', 'median', 'original'
        data_dir: Root data directory containing the episode folders
        start_idx: Starting episode index (inclusive)
        end_idx: Ending episode index (exclusive, None = all)
    """
    print(f"\n{'='*80}")
    print(f"Generating videos for {method_name.upper()} with GLOBAL rank-based quality + min score {MIN_GOOD_SCORE} (episodes {start_idx}-{end_idx or 'end'})")
    print(f"{'='*80}")

    # Track folders missing predict_remain_time
    missing_quality_folders = []

    data_dir_list = os.listdir(data_dir)
    data_dir_list.sort()

    # ========================================================================
    # Phase 1: Collect ALL data from ALL folders (no quality computation yet)
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 1: Collecting all data from ALL folders...")
    print("="*80)

    # Global storage for all episodes across all folders
    global_all_data = []  # Each item: {'episode_file': Path, 'df': DataFrame, 'dir_path': str, 'episode_idx': int}
    global_all_scores = []  # Each item: np.ndarray of scores

    for dir_name in data_dir_list:
        dir_path = os.path.join(data_dir, dir_name)
        method_dir = os.path.join(dir_path, "data", "chunk-000")
        method_dir = Path(method_dir)

        if not method_dir.exists():
            continue

        # Get all episode files
        episode_files = sorted(method_dir.glob('episode_*.parquet'))

        # Apply range filter
        if end_idx is not None:
            episode_files = episode_files[start_idx:end_idx]
        else:
            episode_files = episode_files[start_idx:]

        if len(episode_files) == 0:
            continue

        print(f"  Processing folder: {dir_name} ({len(episode_files)} episodes)")

        for episode_file in episode_files:
            episode_idx = int(episode_file.stem.split('_')[1])

            try:
                df = pd.read_parquet(episode_file)
            except Exception as e:
                print(f"    Error loading {episode_file}: {e}")
                continue

            # Check if predict_remain_time column exists
            if 'predict_remain_time' not in df.columns:
                print(f"    Missing predict_remain_time in {episode_file}, folder: {dir_name}")
                if dir_name not in missing_quality_folders:
                    missing_quality_folders.append(dir_name)
                continue

            prt = df['predict_remain_time'].values

            # Compute chunk scores
            scores = compute_chunk_scores(prt, chunk_size=CHUNK_SIZE)

            global_all_data.append({
                'episode_file': episode_file,
                'df': df,
                'dir_path': dir_path,
                'episode_idx': episode_idx,
                'dir_name': dir_name
            })
            global_all_scores.append(scores)

    print(f"\n  Total episodes collected: {len(global_all_data)}")
    print(f"  Total frames: {sum(len(s) for s in global_all_scores)}")

    if len(global_all_data) == 0:
        print("  No data to process!")
        return

    # ========================================================================
    # Phase 2: Compute quality from GLOBAL ranking across ALL episodes
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: Computing quality from GLOBAL ranking...")
    print("="*80)
    print(f"  Quality thresholds:")
    print(f"    - Top {QUALITY_TOP_PERCENTILE*100:.0f}% (highest scores): GOOD (green) IF score >= {MIN_GOOD_SCORE}")
    print(f"    - Top {QUALITY_TOP_PERCENTILE*100:.0f}% with score < {MIN_GOOD_SCORE}: Demoted to MEDIUM (cyan)")
    print(f"    - Middle {(1-QUALITY_BOTTOM_PERCENTILE-QUALITY_TOP_PERCENTILE)*100:.0f}%: MEDIUM (cyan)")
    print(f"    - Bottom {QUALITY_BOTTOM_PERCENTILE*100:.0f}% (lowest scores): BAD (red)")

    global_all_qualities = compute_quality_from_global_ranking_with_min_score(global_all_scores, min_good_score=MIN_GOOD_SCORE)

    # Phase 2.5: Apply intervention override (after proportional processing)
    print("\n" + "="*80)
    print("PHASE 2.5: Applying human_intervention override...")
    print("="*80)
    global_all_qualities = apply_intervention_override(global_all_qualities, global_all_data)

    # Phase 2.6: Force last LAST_GOOD_FRAMES frames of each episode to GOOD
    print("\n" + "="*80)
    print(f"PHASE 2.6: Forcing last {LAST_GOOD_FRAMES} frames to GOOD...")
    print("="*80)
    last_250_override_count = 0
    for idx, data in enumerate(global_all_data):
        quality = global_all_qualities[idx]
        n = len(quality)
        start = max(0, n - LAST_GOOD_FRAMES)
        for i in range(start, n):
            if quality[i] != 2:
                last_250_override_count += 1
                quality[i] = 2
        global_all_qualities[idx] = quality
    print(f"    Overrode {last_250_override_count} frames to GOOD in last 250 frames")

    # Phase 2.7: Enforce consecutive-frame consistency
    print("\n" + "="*80)
    print(f"PHASE 2.7: Enforcing consecutive-frame consistency (min {MIN_CONSECUTIVE_FRAMES} frames)...")
    print("="*80)
    global_all_qualities = apply_consecutive_consistency(global_all_qualities, min_consecutive=MIN_CONSECUTIVE_FRAMES)

    # Print global statistics
    all_qualities_flat = np.concatenate(global_all_qualities)
    all_scores_flat = np.concatenate(global_all_scores)

    n_good = np.sum(all_qualities_flat == 2)
    n_medium = np.sum(all_qualities_flat == 1)
    n_bad = np.sum(all_qualities_flat == 0)
    total = len(all_qualities_flat)

    # Count how many in top 20% have score < 20
    top_20_pct_threshold_idx = int(len(all_scores_flat) * QUALITY_TOP_PERCENTILE)
    sorted_indices = np.argsort(all_scores_flat)[::-1]
    top_20_percent_scores = all_scores_flat[sorted_indices[:top_20_pct_threshold_idx]]
    demoted_in_top_20 = np.sum(top_20_percent_scores < MIN_GOOD_SCORE)

    print(f"\n  GLOBAL Score statistics:")
    print(f"    - Min score: {np.min(all_scores_flat):.2f}")
    print(f"    - Max score: {np.max(all_scores_flat):.2f}")
    print(f"    - Mean score: {np.mean(all_scores_flat):.2f}")
    print(f"    - Median score: {np.median(all_scores_flat):.2f}")
    print(f"    - Top {QUALITY_TOP_PERCENTILE*100:.0f}% threshold score: {top_20_percent_scores[-1] if len(top_20_percent_scores) > 0 else 0:.2f}")
    print(f"    - In top {QUALITY_TOP_PERCENTILE*100:.0f}% but score < {MIN_GOOD_SCORE}: {demoted_in_top_20}")
    print(f"\n  GLOBAL Quality distribution:")
    print(f"    - GOOD: {n_good} ({n_good/total*100:.1f}%)")
    print(f"    - MEDIUM: {n_medium} ({n_medium/total*100:.1f}%)")
    print(f"    - BAD: {n_bad} ({n_bad/total*100:.1f}%)")

    # ========================================================================
    # Phase 3: Write quality back to parquet files
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 3: Writing quality to parquet files...")
    print("="*80)

    for idx, data in enumerate(global_all_data):
        episode_idx = data['episode_idx']
        df = data['df']
        quality = global_all_qualities[idx]
        episode_file = data['episode_file']

        if idx % 100 == 0 or idx == len(global_all_data) - 1:
            print(f"  Progress: [{idx+1}/{len(global_all_data)}] episode {episode_idx:06d}")

        # Overwrite quality column
        df['quality'] = quality

        # Write back to parquet file
        try:
            df.to_parquet(episode_file, index=False)
        except Exception as e:
            print(f"    Error writing to {episode_file}: {e}")

    print(f"\n  Updated quality in {len(global_all_data)} parquet files")

    # Update meta/info.json for each unique directory
    unique_dir_paths = set(data['dir_path'] for data in global_all_data)
    for dir_path in unique_dir_paths:
        info_path = os.path.join(dir_path, "meta", "info.json")
        if not os.path.exists(info_path):
            print(f"  Warning: {info_path} not found, skipping")
            continue
        with open(info_path, 'r') as f:
            info = json.load(f)
        if 'quality' not in info.get('features', {}):
            info.setdefault('features', {})['quality'] = {
                "dtype": "int64",
                "shape": [1],
                "names": None
            }
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=4, ensure_ascii=False)
            print(f"  Added 'quality' feature to {info_path}")
        else:
            print(f"  'quality' already present in {info_path}")

    # ========================================================================
    # Phase 4: Generate videos (every 50th episode)
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 4: Generating videos (every 50th episode)...")
    print("="*80)

    success_count = 0
    fail_count = 0
    skip_count = 0

    for idx, data in enumerate(global_all_data):
        episode_idx = data['episode_idx']
        df = data['df']
        quality = global_all_qualities[idx]
        dir_path = data['dir_path']
        dir_name = data['dir_name']

        # Only generate video for every 50th episode
        if idx % 20 != 0:
            skip_count += 1
            continue

        print(f"\n  [{idx+1}/{len(global_all_data)}] Generating video for episode {episode_idx:06d} (folder: {dir_name})")

        # Get video paths
        video_paths = get_video_paths(dir_path, episode_idx)

        # Generate output path
        output_video_dir = os.path.join(dir_path, f'{method_name}_filter')
        os.makedirs(output_video_dir, exist_ok=True)
        output_path = os.path.join(output_video_dir, f"episode_{episode_idx:06d}.mp4")

        # Determine which PRT column to use
        prt_column = 'predict_remain_time'

        # Create visualization video
        # success = create_smoothed_visualization_video(
        #     df,
        #     quality,
        #     video_paths,
        #     str(output_path),
        #     prt_column=prt_column,
        #     img_size=256,
        #     chunk_size=50
        # )

        # if success:
        #     success_count += 1
        # else:
        #     fail_count += 1

    print(f"\n  Video generation complete!")
    print(f"    - Success: {success_count}")
    print(f"    - Failed: {fail_count}")
    print(f"    - Skipped: {skip_count}")

    # Write missing quality folders to file
    if missing_quality_folders:
        with open('miss_quality.txt', 'w') as f:
            for folder in missing_quality_folders:
                f.write(f"{folder}\n")
        print(f"\nFolders missing predict_remain_time saved to miss_quality.txt ({len(missing_quality_folders)} folders)")
    else:
        print("\nAll folders have predict_remain_time column.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate rank-based action quality labels (and optional videos) from predict_remain_time."
    )
    parser.add_argument(
        '--data_dir',
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root data directory containing the episode folders "
             f"(default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help="Starting episode index (inclusive).",
    )
    parser.add_argument(
        '--end_idx',
        type=int,
        default=None,
        help="Ending episode index (exclusive). Default: process all episodes.",
    )
    return parser.parse_args()


def main():
    """Main function to generate videos for all methods with rank-based quality and minimum score threshold."""
    args = parse_args()

    print("="*80)
    print("Generating Videos with Rank-Based Quality Assessment + Minimum Score Threshold")
    print("="*80)
    print(f"\nData directory: {args.data_dir}")
    print("\nQuality Assessment Method:")
    print(f"  1. For each 50-frame chunk, compute: sum(prt[t] - prt[t+50])")
    print(f"  2. Rank all chunks globally (lower sum = better)")
    print(f"  3. Top {QUALITY_TOP_PERCENTILE*100:.0f}% = GOOD ONLY IF score >= {MIN_GOOD_SCORE}, else MEDIUM")
    print(f"     Middle = MEDIUM, Bottom {QUALITY_BOTTOM_PERCENTILE*100:.0f}% = BAD")

    # Generate videos for each method
    methods = ['original']

    for method in methods:
        process_videos_for_method(
            method,
            data_dir=args.data_dir,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )


if __name__ == '__main__':
    main()
