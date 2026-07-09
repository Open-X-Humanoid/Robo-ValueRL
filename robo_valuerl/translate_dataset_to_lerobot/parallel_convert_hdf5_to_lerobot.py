#!/usr/bin/env python3
"""
Convert raw HDF5 humanoid trajectories into a LeRobot v2.1-style dataset.

Design goals:
1. Keep one output episode parquet per input trajectory.
2. Encode each camera stream to its own per-episode MP4.
3. Parallelize conversion at the episode level.
4. Avoid LeRobot v3's file-based aggregation writer.

Output layout:
    <output_dir>/
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       └── ...
    ├── videos/
    │   └── chunk-000/
    │       ├── observation.image.image/
    │       │   └── episode_000000.mp4
    │       ├── observation.image.left/
    │       │   └── episode_000000.mp4
    │       └── observation.image.right/
    │           └── episode_000000.mp4
    └── meta/
        ├── info.json
        ├── tasks.jsonl
        ├── episodes.jsonl
        └── episodes_stats.jsonl

Notes:
- This script writes a v2.1-like on-disk structure directly and does NOT call
  LeRobotDataset.create(...).save_episode(), because that path follows v3 rules
  in current LeRobot releases.
- The feature keys are preserved from the original script by default
  (observation.image.*). If your downstream tooling expects the more official
  observation.images.* naming, use --camera_key_style official.
"""


import argparse
import json
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from camera_utils import CameraUtils


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def camera_feature_keys(style: str) -> Tuple[str, str, str]:
    if style == "official":
        return (
            "observation.images.image",
            "observation.images.left",
            "observation.images.right",
        )
    return (
        "observation.image.image",
        "observation.image.left",
        "observation.image.right",
    )


def discover_trajectories(raw_data_dir: Path) -> List[Path]:
    # Prefer recursive search so the script is less sensitive to folder nesting.
    trajs = sorted(raw_data_dir.rglob("trajectory.hdf5"))
    return [p for p in trajs if p.is_file()]


def read_num_steps(trajectory_path: Path) -> int:
    with h5py.File(trajectory_path, "r") as data:
        return int(len(data["master"]["arm_left_position_align"]["data"]))


# -----------------------------
# Stats helpers
# -----------------------------

def _to_list(x: np.ndarray | list | tuple | float | int | bool) -> list:
    arr = np.asarray(x)
    if arr.ndim == 0:
        return [arr.item()]
    return arr.astype(np.float64).reshape(-1).tolist()


def _numeric_stats_from_matrix(matrix: np.ndarray) -> dict:
    # matrix shape: [T, D] or [T]
    arr = np.asarray(matrix)
    if arr.ndim == 1:
        arr = arr[:, None]
    arr = arr.astype(np.float64)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def _init_image_stats() -> dict:
    return {
        "min": np.full((3,), np.inf, dtype=np.float64),
        "max": np.full((3,), -np.inf, dtype=np.float64),
        "sum": np.zeros((3,), dtype=np.float64),
        "sumsq": np.zeros((3,), dtype=np.float64),
        "pixels": 0,
        "frames": 0,
    }


def _update_image_stats(stats: dict, rgb_image: np.ndarray) -> None:
    # rgb_image: HWC uint8. Convert to [0, 1] like LeRobot image stats examples.
    x = rgb_image.astype(np.float64) / 255.0
    reshaped = x.reshape(-1, 3)
    stats["min"] = np.minimum(stats["min"], reshaped.min(axis=0))
    stats["max"] = np.maximum(stats["max"], reshaped.max(axis=0))
    stats["sum"] += reshaped.sum(axis=0)
    stats["sumsq"] += np.square(reshaped).sum(axis=0)
    stats["pixels"] += int(reshaped.shape[0])
    stats["frames"] += 1


def _finalize_image_stats(stats: dict) -> dict:
    pixels = max(int(stats["pixels"]), 1)
    mean = stats["sum"] / pixels
    var = np.maximum(stats["sumsq"] / pixels - np.square(mean), 0.0)
    std = np.sqrt(var)

    def nest(vals: np.ndarray) -> list:
        # Match common LeRobot v2 image stats nesting style: [[[c0]], [[c1]], [[c2]]]
        return [[[float(v)]] for v in vals.tolist()]

    return {
        "min": nest(stats["min"]),
        "max": nest(stats["max"]),
        "mean": nest(mean),
        "std": nest(std),
        "count": [int(stats["frames"])],
    }


# -----------------------------
# Video helpers
# -----------------------------

def make_video_writer(video_path: Path, size_wh: Tuple[int, int], fps: int, codec: str) -> cv2.VideoWriter:
    ensure_dir(video_path.parent)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), size_wh)
    if writer.isOpened():
        return writer

    # Fallback order chosen for practical availability.
    for fallback in ("avc1", "mp4v", "MJPG"):
        if fallback == codec:
            continue
        fourcc = cv2.VideoWriter_fourcc(*fallback)
        writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), size_wh)
        if writer.isOpened():
            return writer

    raise RuntimeError(f"Failed to open VideoWriter for: {video_path}")


# -----------------------------
# Parquet helpers
# -----------------------------

def infer_arrow_array(values: list) -> pa.Array:
    first = None
    for v in values:
        if v is not None:
            first = v
            break

    if first is None:
        return pa.array(values)

    if isinstance(first, bool):
        return pa.array(values, type=pa.bool_())
    if isinstance(first, int) and not isinstance(first, bool):
        return pa.array(values, type=pa.int64())
    if isinstance(first, float):
        return pa.array(values, type=pa.float32())
    if isinstance(first, str):
        return pa.array(values, type=pa.string())

    if isinstance(first, (list, tuple, np.ndarray)):
        arr0 = np.asarray(first)
        if arr0.dtype.kind in ("i", "u"):
            return pa.array([np.asarray(v).reshape(-1).tolist() for v in values], type=pa.list_(pa.int64()))
        if arr0.dtype.kind == "b":
            return pa.array([np.asarray(v).reshape(-1).tolist() for v in values], type=pa.list_(pa.bool_()))
        return pa.array([np.asarray(v, dtype=np.float32).reshape(-1).tolist() for v in values], type=pa.list_(pa.float32()))

    return pa.array(values)


def write_episode_parquet(parquet_path: Path, columns: Dict[str, list]) -> None:
    ensure_dir(parquet_path.parent)
    arrays = {key: infer_arrow_array(values) for key, values in columns.items()}
    table = pa.table(arrays)
    pq.write_table(table, parquet_path, compression="zstd")


# -----------------------------
# Feature / metadata inference
# -----------------------------

def infer_feature_spec(sample_traj: Path, fps: int, camera_keys: Tuple[str, str, str], video_height: int, video_width: int, video_codec: str) -> dict:
    with h5py.File(sample_traj, "r") as data:
        features: dict[str, dict[str, Any]] = {}

        def add_vector_feature(key: str, ds: np.ndarray, dtype: str = "float32") -> None:
            shape = list(np.asarray(ds).shape[1:])
            if len(shape) == 0:
                shape = [1]
            features[key] = {
                "dtype": dtype,
                "shape": shape,
                "names": None,
            }

        main_key, left_key, right_key = camera_keys
        for video_key in (main_key, left_key, right_key):
            features[video_key] = {
                "dtype": "video",
                "shape": [video_height, video_width, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": video_height,
                    "video.width": video_width,
                    "video.channels": 3,
                    "video.codec": video_codec,
                    "video.pix_fmt": "bgr24",
                    "video.is_depth_map": False,
                    "video.fps": fps,
                    "has_audio": False,
                },
            }

        add_vector_feature(
            "observation.state.puppet_left_arm_position",
            data["puppet"]["arm_left_position_align"]["data"],
        )
        add_vector_feature(
            "observation.state.puppet_right_arm_position",
            data["puppet"]["arm_right_position_align"]["data"],
        )

        def feature_from_optional(name: str, fallback_ds: np.ndarray, group: h5py.Group, maybe_key: str) -> None:
            if maybe_key in group.keys():
                add_vector_feature(name, group[maybe_key]["data"])
            else:
                add_vector_feature(name, fallback_ds)

        feature_from_optional(
            "observation.state.puppet_left_gripper_position",
            data["master"]["end_effector_left_position_align"]["data"],
            data["puppet"],
            "end_effector_left_position_align",
        )
        feature_from_optional(
            "observation.state.puppet_right_gripper_position",
            data["master"]["end_effector_right_position_align"]["data"],
            data["puppet"],
            "end_effector_right_position_align",
        )
        add_vector_feature(
            "observation.state.master_left_gripper_position",
            data["master"]["end_effector_left_position_align"]["data"],
        )
        add_vector_feature(
            "observation.state.master_right_gripper_position",
            data["master"]["end_effector_right_position_align"]["data"],
        )
        add_vector_feature(
            "observation.state.master_left_arm_position",
            data["master"]["arm_left_position_align"]["data"],
        )
        add_vector_feature(
            "observation.state.master_right_arm_position",
            data["master"]["arm_right_position_align"]["data"],
        )
        add_vector_feature("action.left_arm_position", data["master"]["arm_left_position_align"]["data"])
        add_vector_feature("action.right_arm_position", data["master"]["arm_right_position_align"]["data"])
        add_vector_feature(
            "action.left_gripper_position",
            data["master"]["end_effector_left_position_align"]["data"],
        )
        add_vector_feature(
            "action.right_gripper_position",
            data["master"]["end_effector_right_position_align"]["data"],
        )

        features["value_reward"] = {"dtype": "float32", "shape": [1], "names": None}
        features["discounted_value_return"] = {"dtype": "float32", "shape": [1], "names": None}
        features["y_pressed"] = {"dtype": "float32", "shape": [1], "names": None}
        features["quality"] = {"dtype": "bool", "shape": [1], "names": None}
        features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
        features["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
        features["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
        features["index"] = {"dtype": "int64", "shape": [1], "names": None}
        features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}

    return features


# -----------------------------
# Core conversion
# -----------------------------

def convert_one_episode(job: dict) -> dict:
    trajectory_path = Path(job["trajectory_path"])
    output_dir = Path(job["output_dir"])
    task_instruction = job["task_instruction"]
    task_index = int(job["task_index"])
    episode_index = int(job["episode_index"])
    global_index_offset = int(job["global_index_offset"])
    fps = int(job["fps"])
    chunk_size = int(job["chunk_size"])
    video_height = int(job["video_height"])
    video_width = int(job["video_width"])
    video_codec = str(job["video_codec"])
    camera_key_style = str(job["camera_key_style"])

    camera_main_key, camera_left_key, camera_right_key = camera_feature_keys(camera_key_style)
    chunk_index = episode_index // chunk_size

    episode_stem = f"episode_{episode_index:06d}"
    parquet_path = output_dir / "data" / f"chunk-{chunk_index:03d}" / f"{episode_stem}.parquet"
    video_main_path = output_dir / "videos" / f"chunk-{chunk_index:03d}" / camera_main_key / f"{episode_stem}.mp4"
    video_left_path = output_dir / "videos" / f"chunk-{chunk_index:03d}" / camera_left_key / f"{episode_stem}.mp4"
    video_right_path = output_dir / "videos" / f"chunk-{chunk_index:03d}" / camera_right_key / f"{episode_stem}.mp4"

    try:
        with h5py.File(trajectory_path, "r") as data:
            simple_data = data["master"]["arm_left_position_align"]["data"]
            timesteps = int(len(simple_data))
            if timesteps <= 0:
                raise ValueError(f"Empty trajectory: {trajectory_path}")

            value_reward = np.ones((timesteps,), dtype=np.float32) * (-0.1)
            if "y_pressed" in data.keys():
                y_pressed = data["y_pressed"][:].astype(np.float32) * 50.0
            else:
                y_pressed = np.zeros((timesteps,), dtype=np.float32)
            y_pressed_cumsum = np.cumsum(y_pressed)

            if "fail" in str(trajectory_path):
                value_reward[-1] = -600.0
            else:
                value_reward[-1] = 100.0

            quality = "suboptimal" not in str(trajectory_path)

            discounted_return = np.zeros((timesteps,), dtype=np.float32)
            discounted_return[-1] = value_reward[-1]
            for i in range(timesteps - 2, -1, -1):
                discounted_return[i] = value_reward[i] + discounted_return[i + 1]
            discounted_return += y_pressed_cumsum.astype(np.float32)

            columns: Dict[str, list] = {
                "observation.state.puppet_left_arm_position": [],
                "observation.state.puppet_right_arm_position": [],
                "observation.state.puppet_left_gripper_position": [],
                "observation.state.puppet_right_gripper_position": [],
                "observation.state.master_left_gripper_position": [],
                "observation.state.master_right_gripper_position": [],
                "observation.state.master_left_arm_position": [],
                "observation.state.master_right_arm_position": [],
                "action.left_arm_position": [],
                "action.right_arm_position": [],
                "action.left_gripper_position": [],
                "action.right_gripper_position": [],
                "value_reward": [],
                "discounted_value_return": [],
                "y_pressed": [],
                "quality": [],
                "timestamp": [],
                "frame_index": [],
                "episode_index": [],
                "index": [],
                "task_index": [],
            }

            img_stats = {
                camera_main_key: _init_image_stats(),
                camera_left_key: _init_image_stats(),
                camera_right_key: _init_image_stats(),
            }

            writer_main = make_video_writer(video_main_path, (video_width, video_height), fps, video_codec)
            writer_left = make_video_writer(video_left_path, (video_width, video_height), fps, video_codec)
            writer_right = make_video_writer(video_right_path, (video_width, video_height), fps, video_codec)

            try:
                for step in range(timesteps):
                    head_image = CameraUtils.decode_color_image(
                        data["camera_observations"]["color_images"]["camera_top"][step],
                        input_format="bgr",
                        output_format="rgb",
                    )
                    left_image = CameraUtils.decode_color_image(
                        data["camera_observations"]["color_images"]["camera_left"][step],
                        input_format="bgr",
                        output_format="rgb",
                    )
                    right_image = CameraUtils.decode_color_image(
                        data["camera_observations"]["color_images"]["camera_right"][step],
                        input_format="bgr",
                        output_format="rgb",
                    )

                    head_image = cv2.resize(head_image, (video_width, video_height), interpolation=cv2.INTER_AREA)
                    left_image = cv2.resize(left_image, (video_width, video_height), interpolation=cv2.INTER_AREA)
                    right_image = cv2.resize(right_image, (video_width, video_height), interpolation=cv2.INTER_AREA)

                    _update_image_stats(img_stats[camera_main_key], head_image)
                    _update_image_stats(img_stats[camera_left_key], left_image)
                    _update_image_stats(img_stats[camera_right_key], right_image)

                    # OpenCV VideoWriter expects BGR.
                    writer_main.write(cv2.cvtColor(head_image, cv2.COLOR_RGB2BGR))
                    writer_left.write(cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR))
                    writer_right.write(cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR))

                    master_left_gripper_position = np.asarray(
                        data["master"]["end_effector_left_position_align"]["data"][step],
                        dtype=np.float32,
                    )
                    master_right_gripper_position = np.asarray(
                        data["master"]["end_effector_right_position_align"]["data"][step],
                        dtype=np.float32,
                    )

                    if "end_effector_left_position_align" in data["puppet"].keys():
                        puppet_left_gripper_position = np.asarray(
                            data["puppet"]["end_effector_left_position_align"]["data"][step],
                            dtype=np.float32,
                        )
                    else:
                        puppet_left_gripper_position = master_left_gripper_position

                    if "end_effector_right_position_align" in data["puppet"].keys():
                        puppet_right_gripper_position = np.asarray(
                            data["puppet"]["end_effector_right_position_align"]["data"][step],
                            dtype=np.float32,
                        )
                    else:
                        puppet_right_gripper_position = master_right_gripper_position

                    columns["observation.state.puppet_left_arm_position"].append(
                        np.asarray(data["puppet"]["arm_left_position_align"]["data"][step], dtype=np.float32).tolist()
                    )
                    columns["observation.state.puppet_right_arm_position"].append(
                        np.asarray(data["puppet"]["arm_right_position_align"]["data"][step], dtype=np.float32).tolist()
                    )
                    columns["observation.state.puppet_left_gripper_position"].append(puppet_left_gripper_position.tolist())
                    columns["observation.state.puppet_right_gripper_position"].append(puppet_right_gripper_position.tolist())
                    columns["observation.state.master_left_gripper_position"].append(master_left_gripper_position.tolist())
                    columns["observation.state.master_right_gripper_position"].append(master_right_gripper_position.tolist())
                    columns["observation.state.master_left_arm_position"].append(
                        np.asarray(data["master"]["arm_left_position_align"]["data"][step], dtype=np.float32).tolist()
                    )
                    columns["observation.state.master_right_arm_position"].append(
                        np.asarray(data["master"]["arm_right_position_align"]["data"][step], dtype=np.float32).tolist()
                    )
                    columns["action.left_arm_position"].append(
                        np.asarray(data["master"]["arm_left_position_align"]["data"][step], dtype=np.float32).tolist()
                    )
                    columns["action.right_arm_position"].append(
                        np.asarray(data["master"]["arm_right_position_align"]["data"][step], dtype=np.float32).tolist()
                    )
                    columns["action.left_gripper_position"].append(master_left_gripper_position.tolist())
                    columns["action.right_gripper_position"].append(master_right_gripper_position.tolist())
                    columns["value_reward"].append(float(value_reward[step]))
                    columns["discounted_value_return"].append(float(discounted_return[step]))
                    columns["y_pressed"].append(float(y_pressed[step]))
                    columns["quality"].append(bool(quality))
                    columns["timestamp"].append(float(step / fps))
                    columns["frame_index"].append(int(step))
                    columns["episode_index"].append(int(episode_index))
                    columns["index"].append(int(global_index_offset + step))
                    columns["task_index"].append(int(task_index))
            finally:
                writer_main.release()
                writer_left.release()
                writer_right.release()

        write_episode_parquet(parquet_path, columns)

        episode_stats = {
            "observation.state.puppet_left_arm_position": _numeric_stats_from_matrix(np.asarray(columns["observation.state.puppet_left_arm_position"], dtype=np.float32)),
            "observation.state.puppet_right_arm_position": _numeric_stats_from_matrix(np.asarray(columns["observation.state.puppet_right_arm_position"], dtype=np.float32)),
            "observation.state.puppet_left_gripper_position": _numeric_stats_from_matrix(np.asarray(columns["observation.state.puppet_left_gripper_position"], dtype=np.float32)),
            "observation.state.puppet_right_gripper_position": _numeric_stats_from_matrix(np.asarray(columns["observation.state.puppet_right_gripper_position"], dtype=np.float32)),
            "observation.state.master_left_gripper_position": _numeric_stats_from_matrix(np.asarray(columns["observation.state.master_left_gripper_position"], dtype=np.float32)),
            "observation.state.master_right_gripper_position": _numeric_stats_from_matrix(np.asarray(columns["observation.state.master_right_gripper_position"], dtype=np.float32)),
            "observation.state.master_left_arm_position": _numeric_stats_from_matrix(np.asarray(columns["observation.state.master_left_arm_position"], dtype=np.float32)),
            "observation.state.master_right_arm_position": _numeric_stats_from_matrix(np.asarray(columns["observation.state.master_right_arm_position"], dtype=np.float32)),
            "action.left_arm_position": _numeric_stats_from_matrix(np.asarray(columns["action.left_arm_position"], dtype=np.float32)),
            "action.right_arm_position": _numeric_stats_from_matrix(np.asarray(columns["action.right_arm_position"], dtype=np.float32)),
            "action.left_gripper_position": _numeric_stats_from_matrix(np.asarray(columns["action.left_gripper_position"], dtype=np.float32)),
            "action.right_gripper_position": _numeric_stats_from_matrix(np.asarray(columns["action.right_gripper_position"], dtype=np.float32)),
            "value_reward": _numeric_stats_from_matrix(np.asarray(columns["value_reward"], dtype=np.float32)),
            "discounted_value_return": _numeric_stats_from_matrix(np.asarray(columns["discounted_value_return"], dtype=np.float32)),
            "y_pressed": _numeric_stats_from_matrix(np.asarray(columns["y_pressed"], dtype=np.float32)),
            "quality": _numeric_stats_from_matrix(np.asarray(columns["quality"], dtype=np.int64)),
            "timestamp": _numeric_stats_from_matrix(np.asarray(columns["timestamp"], dtype=np.float32)),
            "frame_index": _numeric_stats_from_matrix(np.asarray(columns["frame_index"], dtype=np.int64)),
            "episode_index": _numeric_stats_from_matrix(np.asarray(columns["episode_index"], dtype=np.int64)),
            "index": _numeric_stats_from_matrix(np.asarray(columns["index"], dtype=np.int64)),
            "task_index": _numeric_stats_from_matrix(np.asarray(columns["task_index"], dtype=np.int64)),
            camera_main_key: _finalize_image_stats(img_stats[camera_main_key]),
            camera_left_key: _finalize_image_stats(img_stats[camera_left_key]),
            camera_right_key: _finalize_image_stats(img_stats[camera_right_key]),
        }

        return {
            "ok": True,
            "trajectory_path": str(trajectory_path),
            "episode_index": episode_index,
            "length": timesteps,
            "task": task_instruction,
            "parquet_path": str(parquet_path.relative_to(output_dir)),
            "video_paths": [
                str(video_main_path.relative_to(output_dir)),
                str(video_left_path.relative_to(output_dir)),
                str(video_right_path.relative_to(output_dir)),
            ],
            "episode_stats": episode_stats,
        }
    except Exception as exc:  # pragma: no cover - operational path
        return {
            "ok": False,
            "trajectory_path": str(trajectory_path),
            "episode_index": episode_index,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert humanoid HDF5 trajectories to a LeRobot v2.1-style dataset with per-episode parquet/MP4 outputs."
    )
    parser.add_argument("--task_name", type=str, required=True, help="Task name, e.g. separate_the_blocks_and_sort_by_color_into_different_plates")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Directory containing raw trajectory.hdf5 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dataset directory")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Episodes per chunk directory")
    parser.add_argument("--max_workers", type=int, default=max(os.cpu_count() or 1, 1), help="Parallel worker count")
    parser.add_argument("--video_height", type=int, default=336, help="Output video height")
    parser.add_argument("--video_width", type=int, default=336, help="Output video width")
    parser.add_argument("--video_codec", type=str, default="mp4v", help="Preferred fourcc codec, e.g. mp4v or avc1")
    parser.add_argument(
        "--camera_key_style",
        type=str,
        choices=["original", "official"],
        default="original",
        help="original -> observation.image.*, official -> observation.images.*",
    )
    parser.add_argument("--task_index", type=int, default=0, help="Task index written into parquet/tasks.jsonl")
    parser.add_argument("--overwrite", action="store_true", help="Delete output_dir before writing")
    parser.add_argument("--skip_failed", action="store_true", help="Continue even if some trajectories fail")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    raw_data_dir = Path(args.raw_data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not raw_data_dir.exists():
        print(f"[ERROR] raw_data_dir does not exist: {raw_data_dir}", file=sys.stderr)
        return 2

    if output_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(output_dir)

    ensure_dir(output_dir)
    ensure_dir(output_dir / "meta")

    task_instruction = " ".join(args.task_name.split("_"))
    camera_keys = camera_feature_keys(args.camera_key_style)

    traj_list = discover_trajectories(raw_data_dir)
    if not traj_list:
        print(f"[ERROR] No trajectory.hdf5 found under: {raw_data_dir}", file=sys.stderr)
        return 2

    print(f"[INFO] Found {len(traj_list)} trajectories")
    print("[INFO] Pre-scanning lengths for deterministic episode_index/global index assignment...")

    lengths: List[int] = []
    for traj_path in traj_list:
        lengths.append(read_num_steps(traj_path))

    global_offsets: List[int] = []
    running = 0
    for length in lengths:
        global_offsets.append(running)
        running += int(length)

    feature_spec = infer_feature_spec(
        sample_traj=traj_list[0],
        fps=args.fps,
        camera_keys=camera_keys,
        video_height=args.video_height,
        video_width=args.video_width,
        video_codec=args.video_codec,
    )

    jobs: List[dict] = []
    for episode_index, (traj_path, global_offset) in enumerate(zip(traj_list, global_offsets)):
        jobs.append(
            {
                "trajectory_path": str(traj_path),
                "output_dir": str(output_dir),
                "task_instruction": task_instruction,
                "task_index": args.task_index,
                "episode_index": episode_index,
                "global_index_offset": global_offset,
                "fps": args.fps,
                "chunk_size": args.chunk_size,
                "video_height": args.video_height,
                "video_width": args.video_width,
                "video_codec": args.video_codec,
                "camera_key_style": args.camera_key_style,
            }
        )

    results: List[dict] = []
    failures: List[dict] = []

    print(f"[INFO] Converting with {args.max_workers} worker(s)...")
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(convert_one_episode, job) for job in jobs]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result["ok"]:
                results.append(result)
                print(
                    f"[OK] {idx:04d}/{len(jobs):04d} | episode={result['episode_index']:06d} | len={result['length']} | {result['trajectory_path']}"
                )
            else:
                failures.append(result)
                print(
                    f"[FAIL] {idx:04d}/{len(jobs):04d} | episode={result['episode_index']:06d} | {result['trajectory_path']} | {result['error']}",
                    file=sys.stderr,
                )
                if not args.skip_failed:
                    print(result.get("traceback", ""), file=sys.stderr)
                    return 1

    results.sort(key=lambda x: x["episode_index"])

    tasks_rows = [{"task_index": int(args.task_index), "task": task_instruction}]
    episodes_rows = [
        {
            "episode_index": int(r["episode_index"]),
            "tasks": [r["task"]],
            "length": int(r["length"]),
        }
        for r in results
    ]
    episodes_stats_rows = [
        {
            "episode_index": int(r["episode_index"]),
            "stats": r["episode_stats"],
        }
        for r in results
    ]

    total_episodes = len(results)
    total_frames = sum(int(r["length"]) for r in results)
    total_chunks = math.ceil(total_episodes / max(int(args.chunk_size), 1))
    total_videos = total_episodes * 3

    info = {
        "codebase_version": "v2.1",
        "robot_type": "x_humanoid",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": int(args.chunk_size),
        "fps": int(args.fps),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": feature_spec,
    }

    write_json(output_dir / "meta" / "info.json", info)
    write_jsonl(output_dir / "meta" / "tasks.jsonl", tasks_rows)
    write_jsonl(output_dir / "meta" / "episodes.jsonl", episodes_rows)
    write_jsonl(output_dir / "meta" / "episodes_stats.jsonl", episodes_stats_rows)

    print("[INFO] Done")
    print(f"[INFO] Saved dataset to: {output_dir}")
    print(f"[INFO] Successful episodes: {total_episodes}")
    print(f"[INFO] Failed episodes: {len(failures)}")

    if failures:
        failure_path = output_dir / "meta" / "failed_episodes.jsonl"
        write_jsonl(failure_path, failures)
        print(f"[WARN] Failure log written to: {failure_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
