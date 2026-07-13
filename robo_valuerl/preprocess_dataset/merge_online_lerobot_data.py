#!/usr/bin/env python3
"""
Merge LeRobot Datasets within Each Group Folder

For each sub-folder in online_lerobot_data/, find all valid lerobot datasets
(those containing meta/info.json) and merge them into a single dataset.

Structure expected:
  online_lerobot_data/
    <group_folder>/          <- e.g. 16_0226_better_lerobot
      <lerobot_dataset_1>/   <- has meta/info.json
      <lerobot_dataset_2>/
      ...
    <group_folder_2>/
      ...

Output (one merged dataset per group folder):
  output_root/
    <group_folder>_merged/
"""

import argparse
import json
import os
import shutil
import pandas as pd
from typing import List, Dict
# =============================================================================
# Helpers
# =============================================================================

def get_video_keys(dataset_path: str) -> List[str]:
    """Detect video keys from the videos/chunk-000 directory."""
    video_chunk = os.path.join(dataset_path, "videos", "chunk-000")
    if not os.path.isdir(video_chunk):
        return []
    return sorted(os.listdir(video_chunk))


def find_lerobot_datasets(group_path: str) -> List[str]:
    """
    Return sorted list of paths to valid lerobot datasets directly under group_path.
    A valid dataset is a direct child directory that contains meta/info.json.
    """
    datasets = []
    for item in sorted(os.listdir(group_path)):
        candidate = os.path.join(group_path, item)
        if not os.path.isdir(candidate):
            continue
        if os.path.exists(os.path.join(candidate, "meta", "info.json")):
            datasets.append(candidate)
    return datasets


# =============================================================================
# Merge
# =============================================================================

def merge_datasets(
    dataset_paths: List[str],
    output_path: str,
) -> Dict:
    """
    Merge multiple lerobot datasets into output_path.

    Returns a summary dict with episode/frame counts.
    """
    os.makedirs(os.path.join(output_path, "data", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "meta"), exist_ok=True)

    # Detect video keys from the first dataset
    video_keys: List[str] = []
    for dp in dataset_paths:
        vk = get_video_keys(dp)
        if vk:
            video_keys = vk
            break

    for vk in video_keys:
        os.makedirs(os.path.join(output_path, "videos", "chunk-000", vk), exist_ok=True)

    # Load base info.json from the first dataset
    first_info_path = os.path.join(dataset_paths[0], "meta", "info.json")
    with open(first_info_path, "r") as f:
        merged_info = json.load(f)

    new_episode_idx = 0
    cumulative_frame_idx = 0
    total_videos_copied = 0
    merged_episodes: List[dict] = []
    merged_stats: List[dict] = []
    merged_tasks: List[dict] = []
    task_to_index: Dict[str, int] = {}
    current_task_index = 0

    for source_path in dataset_paths:
        source_name = os.path.basename(source_path.rstrip("/"))
        print(f"  -> {source_name}")

        source_data_dir = os.path.join(source_path, "data", "chunk-000")
        source_meta_dir = os.path.join(source_path, "meta")
        source_video_base = os.path.join(source_path, "videos", "chunk-000")

        episodes_path = os.path.join(source_meta_dir, "episodes.jsonl")
        episodes_stats_path = os.path.join(source_meta_dir, "episodes_stats.jsonl")
        tasks_path = os.path.join(source_meta_dir, "tasks.jsonl")

        if not os.path.exists(episodes_path) or not os.path.exists(episodes_stats_path):
            print(f"     [SKIP] missing episodes.jsonl or episodes_stats.jsonl")
            continue

        try:
            episodes_list: List[dict] = []
            with open(episodes_path, "r") as f:
                for line in f:
                    if line.strip():
                        episodes_list.append(json.loads(line))

            episodes_stats_list: List[dict] = []
            with open(episodes_stats_path, "r") as f:
                for line in f:
                    if line.strip():
                        episodes_stats_list.append(json.loads(line))

            if os.path.exists(tasks_path):
                with open(tasks_path, "r") as f:
                    for line in f:
                        if line.strip():
                            task_data = json.loads(line)
                            desc = task_data["task"]
                            if desc not in task_to_index:
                                task_to_index[desc] = current_task_index
                                merged_tasks.append({"task_index": current_task_index, "task": desc})
                                current_task_index += 1
        except Exception as e:
            print(f"     [SKIP] error reading metadata: {e}")
            continue

        for orig_ep_idx in range(len(episodes_list)):
            parquet_file = os.path.join(source_data_dir, f"episode_{orig_ep_idx:06d}.parquet")
            if not os.path.exists(parquet_file):
                print(f"     [SKIP ep {orig_ep_idx}] parquet missing")
                continue

            # Check that all expected video files exist
            missing_video = False
            for vk in video_keys:
                src_video = os.path.join(source_video_base, vk, f"episode_{orig_ep_idx:06d}.mp4")
                if not os.path.exists(src_video):
                    print(f"     [SKIP ep {orig_ep_idx}] missing video {vk}/episode_{orig_ep_idx:06d}.mp4")
                    missing_video = True
                    break
            if missing_video:
                continue

            # Read and re-index parquet
            try:
                df = pd.read_parquet(parquet_file)
            except Exception as e:
                print(f"     [SKIP ep {orig_ep_idx}] corrupted parquet: {e}")
                continue

            num_rows = len(df)
            df["episode_index"] = new_episode_idx
            df["index"] = list(range(cumulative_frame_idx, cumulative_frame_idx + num_rows))
            df["frame_index"] = list(range(num_rows))

            merged_parquet = os.path.join(
                output_path, "data", "chunk-000", f"episode_{new_episode_idx:06d}.parquet"
            )
            df.to_parquet(merged_parquet)

            # Copy videos
            for vk in video_keys:
                src_video = os.path.join(source_video_base, vk, f"episode_{orig_ep_idx:06d}.mp4")
                dst_video = os.path.join(
                    output_path, "videos", "chunk-000", vk, f"episode_{new_episode_idx:06d}.mp4"
                )
                shutil.copy2(src_video, dst_video)
                total_videos_copied += 1

            # Episode metadata
            ep_info = episodes_list[orig_ep_idx].copy()
            ep_info["episode_index"] = new_episode_idx
            merged_episodes.append(ep_info)

            ep_stats = episodes_stats_list[orig_ep_idx].copy()
            ep_stats["episode_index"] = new_episode_idx
            merged_stats.append(ep_stats)

            cumulative_frame_idx += num_rows
            new_episode_idx += 1

    # Write merged metadata
    with open(os.path.join(output_path, "meta", "episodes.jsonl"), "w") as f:
        for item in merged_episodes:
            f.write(json.dumps(item) + "\n")

    with open(os.path.join(output_path, "meta", "episodes_stats.jsonl"), "w") as f:
        for item in merged_stats:
            f.write(json.dumps(item) + "\n")

    if merged_tasks:
        with open(os.path.join(output_path, "meta", "tasks.jsonl"), "w") as f:
            for item in merged_tasks:
                f.write(json.dumps(item) + "\n")

    # Update info.json
    merged_info["total_episodes"] = len(merged_episodes)
    merged_info["total_frames"] = cumulative_frame_idx
    merged_info["total_videos"] = total_videos_copied

    with open(os.path.join(output_path, "meta", "info.json"), "w") as f:
        json.dump(merged_info, f, indent=4)

    return {
        "total_episodes": len(merged_episodes),
        "total_frames": cumulative_frame_idx,
        "total_videos": total_videos_copied,
    }


# =============================================================================
# Main
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Merge lerobot datasets within each group folder in online_lerobot_data"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/dataset/toago6/workspace/code/rl_block_x_humanoid_data/online_lerobot_data",
        help="Root directory containing group folders",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output directory for merged datasets (default: <data_root>/../online_lerobot_data_merged)",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Specific group folder names to process (default: all)",
    )
    return parser.parse_args()


def main():
    args = get_args()

    data_root = os.path.abspath(args.data_root)
    if args.output_root is None:
        output_root = os.path.join(os.path.dirname(data_root), "online_lerobot_data_merged")
    else:
        output_root = os.path.abspath(args.output_root)

    os.makedirs(output_root, exist_ok=True)

    print("=" * 60)
    print("LeRobot Online Data Merge")
    print("=" * 60)
    print(f"Data root   : {data_root}")
    print(f"Output root : {output_root}")
    print("=" * 60)

    # Collect group folders
    if args.groups:
        group_names = args.groups
    else:
        group_names = sorted(
            item for item in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, item))
        )

    overall_summary = []

    for group_name in group_names:
        group_path = os.path.join(data_root, group_name)
        if not os.path.isdir(group_path):
            print(f"\n[WARNING] Group not found: {group_name}")
            continue

        datasets = find_lerobot_datasets(group_path)
        if not datasets:
            print(f"\n[SKIP] {group_name}: no valid lerobot datasets found")
            continue

        output_name = f"{group_name}_merged"
        output_path = os.path.join(output_root, output_name)

        print(f"\n{'=' * 60}")
        print(f"Group: {group_name}")
        print(f"  Datasets found: {len(datasets)}")
        print(f"  Output: {output_name}")
        print(f"{'=' * 60}")

        if os.path.exists(output_path):
            print(f"  [WARNING] Output already exists, removing: {output_path}")
            shutil.rmtree(output_path)

        summary = merge_datasets(datasets, output_path)
        summary["group"] = group_name
        overall_summary.append(summary)

        print(f"  Done: {summary['total_episodes']} episodes, "
              f"{summary['total_frames']} frames, "
              f"{summary['total_videos']} videos")

    print(f"\n{'=' * 60}")
    print("MERGE COMPLETE")
    print("=" * 60)
    for s in overall_summary:
        print(f"  {s['group']:40s}  episodes={s['total_episodes']:4d}  frames={s['total_frames']:6d}")
    print(f"\nTotal groups merged: {len(overall_summary)}")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
