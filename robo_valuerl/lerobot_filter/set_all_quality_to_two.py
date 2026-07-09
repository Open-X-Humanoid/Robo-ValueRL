#!/usr/bin/env python3
"""
Set all quality fields to int64 with value 1 in all LeRobot datasets.

This script processes all lerobot datasets in a specified directory,
modifying the quality field in each episode to be int64 type with value 1.
"""

import json
import os
from typing import Dict

import numpy as np
import pandas as pd

# Quality feature definition
QUALITY_FEATURE = {
    "quality": {
        "dtype": "int64",
        "shape": [1],
        "names": None
    }
}


def set_quality_to_one(dataset_path: str) -> Dict[str, int]:
    """
    Set quality field to int64 with value 1 for all episodes in a dataset.

    Args:
        dataset_path: Path to lerobot dataset

    Returns:
        Dictionary with processing statistics
    """
    data_dir = os.path.join(dataset_path, "data/chunk-000")
    info_path = os.path.join(dataset_path, "meta/info.json")

    stats = {
        "episodes_processed": 0,
        "total_frames": 0
    }

    # Update info.json to include quality feature
    with open(info_path, 'r') as f:
        info_data = json.load(f)

    if "features" not in info_data:
        info_data["features"] = {}
    info_data["features"].update(QUALITY_FEATURE)

    with open(info_path, 'w') as f:
        json.dump(info_data, f, indent=4)

    # Process all parquet files
    parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet")])

    for parquet_file in parquet_files:
        parquet_path = os.path.join(data_dir, parquet_file)

        # Read parquet file
        df = pd.read_parquet(parquet_path)
        num_rows = len(df)

        # Set quality to int64 with value 1
        df['quality'] = np.array([2] * num_rows, dtype=np.int64)

        # Write back
        df.to_parquet(parquet_path)

        stats["episodes_processed"] += 1
        stats["total_frames"] += num_rows

    return stats


def main():
    # Target directory containing all lerobot datasets
    datasets_root = "/mnt/dataset/toago6/workspace/code/rl_block_x_humanoid_data/stack_0520_after/lerobot_data"

    print("=" * 60)
    print("Setting all quality fields to int64 with value 2")
    print("=" * 60)
    print(f"Target directory: {datasets_root}")
    print("=" * 60)

    total_datasets = 0
    total_episodes = 0
    total_frames = 0

    # Iterate through all items in the directory
    for item in sorted(os.listdir(datasets_root)):
        dataset_path = os.path.join(datasets_root, item)

        # Skip if not a directory
        if not os.path.isdir(dataset_path):
            continue

        # Check if it's a valid lerobot dataset
        info_path = os.path.join(dataset_path, "meta/info.json")
        if not os.path.exists(info_path):
            continue

        total_datasets += 1
        print(f"\n[{total_datasets}] Processing: {item}")

        try:
            stats = set_quality_to_one(dataset_path)
            print(f"  Episodes: {stats['episodes_processed']}, Frames: {stats['total_frames']}")
            total_episodes += stats['episodes_processed']
            total_frames += stats['total_frames']
        except Exception as e:
            print(f"  [ERROR] Failed to process: {e}")
            continue

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Datasets processed: {total_datasets}")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total frames: {total_frames}")
    print("=" * 60)
    print("All quality fields have been set to int64 with value 1!")


if __name__ == "__main__":
    main()
