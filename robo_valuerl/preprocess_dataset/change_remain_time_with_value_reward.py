import argparse
import json
import os
import math
import numpy as np
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description="Update/create remain_time based on last frame's value_reward")
    parser.add_argument("--root", type=str, required=True, help="Input dataset root path")
    parser.add_argument("--offset", type=float, default=2500, help="Constant to add to remain_time when value_reward != 0")
    return parser.parse_args()


def main():
    args = get_args()

    for filename in os.listdir(args.root):
        file_path = os.path.join(args.root, filename)
        info_path = os.path.join(file_path, "meta", "info.json")

        if not os.path.exists(info_path):
            print(f"Error: could not find info.json at {info_path}")
            return

        with open(info_path, 'r') as f:
            info_data = json.load(f)

        fps = info_data.get('fps', 30)
        time_scale = math.ceil(30 / fps)  # When fps=30, time_scale=1

        total_episodes = info_data.get('total_episodes', 0)
        print(f"Starting to update dataset {filename}, fps={fps}, time_scale={time_scale}, {total_episodes} episodes total...")

        # Iterate over all chunks
        data_base_dir = os.path.join(file_path, "data")
        chunk_dirs = sorted([d for d in os.listdir(data_base_dir) if d.startswith("chunk-")])

        for chunk_dir in chunk_dirs:
            data_dir = os.path.join(data_base_dir, chunk_dir)
            print(f"\nProcessing {chunk_dir}...")

            parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet")])

            for parquet_filename in parquet_files:
                parquet_file = os.path.join(data_dir, parquet_filename)

                if not os.path.exists(parquet_file):
                    print(f"Skipping: file does not exist {parquet_file}")
                    continue

                # 1. Read the Parquet file
                df = pd.read_parquet(parquet_file)

                # 2. Check the value_reward of the last frame
                if 'value_reward' not in df.columns:
                    print(f"Skipping {parquet_filename}: no value_reward column")
                    continue

                last_value_reward = df['value_reward'].iloc[-1]
                T = len(df)

                # 3. Compute remain_time
                # remain_time[t] = (T - t) * time_scale
                remain_time = np.array([(T - t) * time_scale for t in range(T)], dtype=np.float32)

                # 4. Decide whether to add an offset based on the last frame's value_reward
                # Only add the offset when value_reward is neither 0 nor 100
                if last_value_reward != 0 and last_value_reward != 100:
                    remain_time = remain_time + args.offset
                    print(f"{parquet_filename}: last frame value_reward={last_value_reward}, remain_time increased by {args.offset}")
                else:
                    print(f"{parquet_filename}: last frame value_reward={last_value_reward}, remain_time computed normally")

                # 5. Write back (add/modify the remain_time and return_to_go columns)
                df['remain_time'] = remain_time
                df['return_to_go'] = np.zeros(T, dtype=np.float32)
                df.to_parquet(parquet_file)

        # 6. Update the metadata in info.json
        if "features" not in info_data:
            info_data["features"] = {}
        info_data["features"]["remain_time"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None
        }
        info_data["features"]["return_to_go"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None
        }

        with open(info_path, 'w') as f:
            json.dump(info_data, f, indent=4)

        print(f"\n[Done] remain_time has been updated for all episodes, and info.json has been updated.")


if __name__ == "__main__":
    main()
