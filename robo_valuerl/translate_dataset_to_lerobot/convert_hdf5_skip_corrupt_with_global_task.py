#!/usr/bin/env python3
"""
Convert raw HDF5 humanoid trajectories into LeRobot v2.1-style datasets.

This version supports two modes:
1. Single-task mode:
   python script.py --raw_data_dir /path/to/task_dir --output_dir /path/out/task_x --task_name task_x
2. Multi-task auto-discovery mode:
   python script.py --raw_data_dir /path/to/raw_root --output_dir /path/out_root

In multi-task mode, the script treats the first N relative path components under
raw_data_dir as the task key (N is controlled by --task_dir_depth, default 1).
Each discovered task is exported to its own independent LeRobot dataset:

    <output_dir>/
    ├── task_a/
    │   ├── data/
    │   ├── videos/
    │   └── meta/
    ├── task_b/
    │   ├── data/
    │   ├── videos/
    │   └── meta/
    └── conversion_summary.json

Per input trajectory, the converter writes exactly one:
- data/chunk-xxx/episode_xxxxxx.parquet
- videos/chunk-xxx/<camera_key>/episode_xxxxxx.mp4

It does NOT use LeRobot v3's aggregation writer.

Subgoal support:
- Use --annotation_json to provide the annotation json file.
- The script expands subgoal intervals into per-frame fields:
    subgoal_index, subgoal_text, subgoal_progress,
    is_subgoal_start, is_subgoal_end, has_subgoal
- The script also writes global-task fields for every frame:
    global_task_index, global_task_text
- If --use_subgoal_as_task is enabled, each frame's task_index will be replaced
  by its current subgoal_index, and meta/tasks.jsonl will be written from the
  subgoal vocabulary. This is useful when later training code mainly consumes
  task_index/task text.
- Even when --use_subgoal_as_task is enabled, the original global task is still
  preserved in parquet/meta so downstream code can read both global task and
  subgoal.
"""


import argparse
import json
import math
import os
import shutil
import sys
import traceback
from collections import defaultdict
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


def safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def is_probably_corrupted_hdf5_error(exc: Exception, tb_text: str = "") -> bool:
    if isinstance(exc, (OSError, EOFError)):
        return True

    msg = f"{type(exc).__name__}: {exc}".lower()
    tb = (tb_text or "").lower()
    patterns = [
        "file signature not found",
        "truncated file",
        "bad object header",
        "unable to open file",
        "unable to synchronously open",
        "unable to synchronously read",
        "can't read data",
        "wrong b-tree signature",
        "wrong fractal heap header signature",
        "inflate() failed",
        "filter returned failure",
        "addr overflow",
        "object header message is too large",
        "h5dread failed",
        "h5fopen failed",
    ]
    return any(p in msg or p in tb for p in patterns)


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


def sanitize_task_output_name(task_key: str) -> str:
    parts = [p for p in task_key.replace("\\", "/").split("/") if p]
    if not parts:
        return "root_task"
    return "__".join(parts)


def task_key_to_instruction(task_key: str) -> str:
    return task_key.replace("/", " ").replace("_", " ").strip()


def resolve_global_task_instruction(task_key: str) -> str:
    normalized = str(task_key).lower().replace("_", "").replace("-", "").replace("/", "").replace(" ", "")
    if "electricmeter" in normalized:
        return "Inspect joint quality and separate good parts from defective parts"
    return task_key_to_instruction(task_key)


def discover_trajectories(raw_data_dir: Path) -> List[Path]:
    trajs = sorted(raw_data_dir.rglob("trajectory.hdf5"))
    return [p for p in trajs if p.is_file()]


def discover_task_groups(raw_data_dir: Path, task_name: str | None, task_dir_depth: int) -> Dict[str, List[Path]]:
    trajs = discover_trajectories(raw_data_dir)
    if not trajs:
        return {}

    if task_name:
        return {task_name: trajs}

    groups: Dict[str, List[Path]] = defaultdict(list)
    for traj in trajs:
        rel = traj.relative_to(raw_data_dir)
        parts = list(rel.parts)
        if len(parts) <= 1:
            task_key = raw_data_dir.name
        else:
            depth = max(int(task_dir_depth), 1)
            usable = parts[:-1]
            task_parts = usable[:depth] if len(usable) >= depth else usable
            task_key = "/".join(task_parts) if task_parts else raw_data_dir.name
        groups[task_key].append(traj)

    return {k: sorted(v) for k, v in sorted(groups.items(), key=lambda kv: kv[0])}


def read_num_steps(trajectory_path: Path) -> int:
    with h5py.File(trajectory_path, "r") as data:
        return int(len(data["master"]["arm_left_position_align"]["data"]))


def read_num_steps_safe(trajectory_path: Path) -> tuple[bool, int | None, dict | None]:
    try:
        return True, read_num_steps(trajectory_path), None
    except Exception as exc:
        tb_text = traceback.format_exc()
        return False, None, {
            "trajectory_path": str(trajectory_path),
            "stage": "pre_scan",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": tb_text,
            "is_probably_corrupted_hdf5": is_probably_corrupted_hdf5_error(exc, tb_text),
        }


# -----------------------------
# Annotation / subgoal helpers
# -----------------------------

def split_path_parts(path_like: str | Path) -> list[str]:
    return [p for p in str(path_like).replace("\\", "/").split("/") if p]


def get_episode_key_from_any_path(path_like: str | Path) -> str:
    """
    Extract episode id such as 0326_150036 from either:
    - annotation rawDataPath .../success_episodes/<episode_id>/data
    - actual hdf5 path .../success_episodes/<episode_id>/data/trajectory.hdf5
    """
    parts = split_path_parts(path_like)

    if "success_episodes" in parts:
        idx = parts.index("success_episodes")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    if len(parts) >= 3 and parts[-1] == "trajectory.hdf5" and parts[-2] == "data":
        return parts[-3]

    if len(parts) >= 2 and parts[-1] == "data":
        return parts[-2]

    stem = parts[-1] if parts else ""
    if stem.endswith(".hdf5"):
        stem = stem[:-5]
    return stem


def load_annotation_index(annotation_json_path: str | None) -> dict[str, dict]:
    if not annotation_json_path:
        return {}

    ann_path = Path(annotation_json_path)
    ann_index: dict[str, dict] = {}

    if ann_path.is_dir():
        json_files = sorted(ann_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No json files found under annotation directory: {ann_path}"
            )
    elif ann_path.is_file():
        json_files = [ann_path]
    else:
        raise FileNotFoundError(
            f"Annotation path does not exist: {annotation_json_path}"
        )

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            obj = json.load(f)

        episode_labels = obj.get("data", {}).get("episodeLabels", [])
        for ep in episode_labels:
            raw_data_path = ep.get("rawDataPath", "")
            episode_key = get_episode_key_from_any_path(raw_data_path)
            if episode_key:
                ann_index[episode_key] = ep

    return ann_index


def collect_subgoal_vocab_for_trajs(traj_list: list[Path], ann_index: dict[str, dict]) -> tuple[list[str], dict[str, int]]:
    """
    Build a stable subgoal vocabulary for one task dataset.
    Index 0 is reserved for <none> so unlabeled frames are valid.
    """
    vocab = ["<none>"]
    seen = {"<none>"}

    for traj in traj_list:
        ep_key = get_episode_key_from_any_path(traj)
        ann = ann_index.get(ep_key)
        if not ann:
            continue
        for item in ann.get("labels", []):
            label = str(item.get("label", "")).strip()
            if label and label not in seen:
                seen.add(label)
                vocab.append(label)

    return vocab, {name: idx for idx, name in enumerate(vocab)}


def parse_frame_range(frame_str: str) -> tuple[int, int]:
    s, e = [x.strip() for x in str(frame_str).split(",", 1)]
    return int(s), int(e)


def build_framewise_subgoal(
    num_frames: int,
    annotation_record: dict | None,
    subgoal_to_idx: dict[str, int],
) -> dict[str, Any]:
    """
    Expand interval annotations into frame-wise arrays.
    Later segments override earlier ones if there is overlap.
    """
    subgoal_index = np.zeros((num_frames,), dtype=np.int64)
    subgoal_text = np.array(["<none>"] * num_frames, dtype=object)
    subgoal_progress = np.zeros((num_frames,), dtype=np.float32)
    is_subgoal_start = np.zeros((num_frames,), dtype=bool)
    is_subgoal_end = np.zeros((num_frames,), dtype=bool)
    has_subgoal = np.zeros((num_frames,), dtype=bool)

    episode_subgoals: list[str] = []

    if not annotation_record:
        return {
            "subgoal_index": subgoal_index,
            "subgoal_text": subgoal_text,
            "subgoal_progress": subgoal_progress,
            "is_subgoal_start": is_subgoal_start,
            "is_subgoal_end": is_subgoal_end,
            "has_subgoal": has_subgoal,
            "episode_subgoals": ["<none>"],
        }

    seen_episode_subgoals = set()
    for item in annotation_record.get("labels", []):
        label_name = str(item.get("label", "")).strip()
        if not label_name or label_name not in subgoal_to_idx:
            continue

        try:
            start_f, end_f = parse_frame_range(item["frame"])
        except Exception:
            continue

        start_f = max(0, int(start_f))
        end_f = min(num_frames - 1, int(end_f))
        if end_f < start_f:
            continue

        sg_idx = int(subgoal_to_idx[label_name])
        denom = max(end_f - start_f, 1)

        subgoal_index[start_f:end_f + 1] = sg_idx
        subgoal_text[start_f:end_f + 1] = label_name
        has_subgoal[start_f:end_f + 1] = True
        is_subgoal_start[start_f] = True
        is_subgoal_end[end_f] = True

        for f in range(start_f, end_f + 1):
            subgoal_progress[f] = (f - start_f) / denom

        if label_name not in seen_episode_subgoals:
            seen_episode_subgoals.add(label_name)
            episode_subgoals.append(label_name)

    if not episode_subgoals:
        episode_subgoals = ["<none>"]

    return {
        "subgoal_index": subgoal_index,
        "subgoal_text": subgoal_text,
        "subgoal_progress": subgoal_progress,
        "is_subgoal_start": is_subgoal_start,
        "is_subgoal_end": is_subgoal_end,
        "has_subgoal": has_subgoal,
        "episode_subgoals": episode_subgoals,
    }


CAMERA_ALIASES = {
    "main": ("camera_head", "camera_top"),
    "left": ("camera_left", "left_camera"),
    "right": ("camera_right", "right_camera"),
}


def resolve_camera_dataset(color_images_group: h5py.Group, role: str) -> tuple[h5py.Dataset, str]:
    aliases = CAMERA_ALIASES[role]
    for key in aliases:
        if key in color_images_group:
            return color_images_group[key], key

    available = sorted(list(color_images_group.keys()))
    raise KeyError(
        f"Missing {role} camera. Tried aliases={list(aliases)}, available={available}"
    )


def resolve_camera_datasets(data: h5py.File) -> dict[str, tuple[h5py.Dataset, str]]:
    if "camera_observations" not in data:
        raise KeyError("Missing group: camera_observations")
    camera_obs = data["camera_observations"]
    if "color_images" not in camera_obs:
        raise KeyError("Missing group: camera_observations/color_images")
    color_images = camera_obs["color_images"]

    return {
        "main": resolve_camera_dataset(color_images, "main"),
        "left": resolve_camera_dataset(color_images, "left"),
        "right": resolve_camera_dataset(color_images, "right"),
    }


# -----------------------------
# Stats helpers
# -----------------------------

def _numeric_stats_from_matrix(matrix: np.ndarray) -> dict:
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
        return pa.array(
            [np.asarray(v, dtype=np.float32).reshape(-1).tolist() for v in values],
            type=pa.list_(pa.float32()),
        )

    return pa.array(values)


def write_episode_parquet(parquet_path: Path, columns: Dict[str, list]) -> None:
    ensure_dir(parquet_path.parent)
    arrays = {key: infer_arrow_array(values) for key, values in columns.items()}
    table = pa.table(arrays)
    pq.write_table(table, parquet_path, compression="zstd")


# -----------------------------
# Feature / metadata inference
# -----------------------------

def infer_feature_spec(
    sample_traj: Path,
    fps: int,
    camera_keys: Tuple[str, str, str],
    video_height: int,
    video_width: int,
    video_codec: str,
) -> dict:
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
        features["global_task_index"] = {"dtype": "int64", "shape": [1], "names": None}
        features["global_task_text"] = {"dtype": "string", "shape": [1], "names": None}

        # Subgoal-related per-frame features
        features["subgoal_index"] = {"dtype": "int64", "shape": [1], "names": None}
        features["subgoal_text"] = {"dtype": "string", "shape": [1], "names": None}
        features["subgoal_progress"] = {"dtype": "float32", "shape": [1], "names": None}
        features["is_subgoal_start"] = {"dtype": "bool", "shape": [1], "names": None}
        features["is_subgoal_end"] = {"dtype": "bool", "shape": [1], "names": None}
        features["has_subgoal"] = {"dtype": "bool", "shape": [1], "names": None}

    return features


def infer_feature_spec_safe(
    sample_traj: Path,
    fps: int,
    camera_keys: Tuple[str, str, str],
    video_height: int,
    video_width: int,
    video_codec: str,
) -> tuple[bool, dict | None, dict | None]:
    try:
        spec = infer_feature_spec(
            sample_traj=sample_traj,
            fps=fps,
            camera_keys=camera_keys,
            video_height=video_height,
            video_width=video_width,
            video_codec=video_codec,
        )
        return True, spec, None
    except Exception as exc:
        tb_text = traceback.format_exc()
        return False, None, {
            "trajectory_path": str(sample_traj),
            "stage": "infer_feature_spec",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": tb_text,
            "is_probably_corrupted_hdf5": is_probably_corrupted_hdf5_error(exc, tb_text),
        }


# -----------------------------
# Core conversion
# -----------------------------

def convert_one_episode(job: dict) -> dict:
    trajectory_path = Path(job["trajectory_path"])
    output_dir = Path(job["output_dir"])
    task_instruction = job["task_instruction"]
    task_output_name = job["task_output_name"]
    task_index = int(job["task_index"])
    global_task_index = int(job.get("global_task_index", task_index))
    global_task_text = str(job.get("global_task_text", task_instruction))
    episode_index = int(job["episode_index"])
    global_index_offset = int(job["global_index_offset"])
    fps = int(job["fps"])
    chunk_size = int(job["chunk_size"])
    video_height = int(job["video_height"])
    video_width = int(job["video_width"])
    video_codec = str(job["video_codec"])
    camera_key_style = str(job["camera_key_style"])
    annotation_record = job.get("annotation_record", None)
    subgoal_to_idx = job.get("subgoal_to_idx", {"<none>": 0})
    use_subgoal_as_task = bool(job.get("use_subgoal_as_task", False))

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

            framewise_subgoal = build_framewise_subgoal(
                num_frames=timesteps,
                annotation_record=annotation_record,
                subgoal_to_idx=subgoal_to_idx,
            )

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
                "global_task_index": [],
                "global_task_text": [],
                "subgoal_index": [],
                "subgoal_text": [],
                "subgoal_progress": [],
                "is_subgoal_start": [],
                "is_subgoal_end": [],
                "has_subgoal": [],
            }

            img_stats = {
                camera_main_key: _init_image_stats(),
                camera_left_key: _init_image_stats(),
                camera_right_key: _init_image_stats(),
            }

            resolved_cameras = resolve_camera_datasets(data)
            main_camera_ds, main_camera_source = resolved_cameras["main"]
            left_camera_ds, left_camera_source = resolved_cameras["left"]
            right_camera_ds, right_camera_source = resolved_cameras["right"]

            writer_main = make_video_writer(video_main_path, (video_width, video_height), fps, video_codec)
            writer_left = make_video_writer(video_left_path, (video_width, video_height), fps, video_codec)
            writer_right = make_video_writer(video_right_path, (video_width, video_height), fps, video_codec)

            try:
                for step in range(timesteps):
                    head_image = CameraUtils.decode_color_image(
                        main_camera_ds[step],
                        input_format="bgr",
                        output_format="rgb",
                    )
                    left_image = CameraUtils.decode_color_image(
                        left_camera_ds[step],
                        input_format="bgr",
                        output_format="rgb",
                    )
                    right_image = CameraUtils.decode_color_image(
                        right_camera_ds[step],
                        input_format="bgr",
                        output_format="rgb",
                    )

                    head_image = cv2.resize(head_image, (video_width, video_height), interpolation=cv2.INTER_AREA)
                    left_image = cv2.resize(left_image, (video_width, video_height), interpolation=cv2.INTER_AREA)
                    right_image = cv2.resize(right_image, (video_width, video_height), interpolation=cv2.INTER_AREA)

                    _update_image_stats(img_stats[camera_main_key], head_image)
                    _update_image_stats(img_stats[camera_left_key], left_image)
                    _update_image_stats(img_stats[camera_right_key], right_image)

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

                    current_subgoal_index = int(framewise_subgoal["subgoal_index"][step])
                    current_subgoal_text = str(framewise_subgoal["subgoal_text"][step])
                    current_subgoal_progress = float(framewise_subgoal["subgoal_progress"][step])
                    current_is_subgoal_start = bool(framewise_subgoal["is_subgoal_start"][step])
                    current_is_subgoal_end = bool(framewise_subgoal["is_subgoal_end"][step])
                    current_has_subgoal = bool(framewise_subgoal["has_subgoal"][step])
                    effective_task_index = current_subgoal_index if use_subgoal_as_task else int(task_index)

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
                    columns["task_index"].append(int(effective_task_index))
                    columns["global_task_index"].append(int(global_task_index))
                    columns["global_task_text"].append(global_task_text)
                    columns["subgoal_index"].append(current_subgoal_index)
                    columns["subgoal_text"].append(current_subgoal_text)
                    columns["subgoal_progress"].append(current_subgoal_progress)
                    columns["is_subgoal_start"].append(current_is_subgoal_start)
                    columns["is_subgoal_end"].append(current_is_subgoal_end)
                    columns["has_subgoal"].append(current_has_subgoal)
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
            "global_task_index": _numeric_stats_from_matrix(np.asarray(columns["global_task_index"], dtype=np.int64)),
            "subgoal_index": _numeric_stats_from_matrix(np.asarray(columns["subgoal_index"], dtype=np.int64)),
            "subgoal_progress": _numeric_stats_from_matrix(np.asarray(columns["subgoal_progress"], dtype=np.float32)),
            "is_subgoal_start": _numeric_stats_from_matrix(np.asarray(columns["is_subgoal_start"], dtype=np.int64)),
            "is_subgoal_end": _numeric_stats_from_matrix(np.asarray(columns["is_subgoal_end"], dtype=np.int64)),
            "has_subgoal": _numeric_stats_from_matrix(np.asarray(columns["has_subgoal"], dtype=np.int64)),
            camera_main_key: _finalize_image_stats(img_stats[camera_main_key]),
            camera_left_key: _finalize_image_stats(img_stats[camera_left_key]),
            camera_right_key: _finalize_image_stats(img_stats[camera_right_key]),
        }

        episode_task_names = framewise_subgoal["episode_subgoals"] if use_subgoal_as_task else [task_instruction]
        episode_subgoal_names = framewise_subgoal["episode_subgoals"]

        return {
            "ok": True,
            "trajectory_path": str(trajectory_path),
            "task_output_name": task_output_name,
            "task": task_instruction,
            "global_task": global_task_text,
            "global_task_index": global_task_index,
            "episode_task_names": episode_task_names,
            "episode_subgoal_names": episode_subgoal_names,
            "episode_index": episode_index,
            "length": timesteps,
            "parquet_path": str(parquet_path.relative_to(output_dir)),
            "video_paths": [
                str(video_main_path.relative_to(output_dir)),
                str(video_left_path.relative_to(output_dir)),
                str(video_right_path.relative_to(output_dir)),
            ],
            "camera_sources": {
                "main": main_camera_source,
                "left": left_camera_source,
                "right": right_camera_source,
            },
            "episode_stats": episode_stats,
        }
    except Exception as exc:
        tb_text = traceback.format_exc()
        safe_unlink(parquet_path)
        safe_unlink(video_main_path)
        safe_unlink(video_left_path)
        safe_unlink(video_right_path)
        return {
            "ok": False,
            "trajectory_path": str(trajectory_path),
            "task_output_name": task_output_name,
            "episode_index": episode_index,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": tb_text,
            "is_probably_corrupted_hdf5": is_probably_corrupted_hdf5_error(exc, tb_text),
        }


# -----------------------------
# Metadata writing
# -----------------------------

def write_task_dataset_meta(
    dataset_output_dir: Path,
    task_instruction: str,
    global_task_index: int,
    results: List[dict],
    feature_spec: dict,
    chunk_size: int,
    fps: int,
    failures: List[dict],
    subgoal_vocab: List[str],
    use_subgoal_as_task: bool,
) -> None:
    results = sorted(results, key=lambda x: x["episode_index"])

    if use_subgoal_as_task:
        tasks_rows = [{"task_index": i, "task": name} for i, name in enumerate(subgoal_vocab)]
    else:
        tasks_rows = [{"task_index": int(global_task_index), "task": task_instruction}]

    global_tasks_rows = [{"global_task_index": int(global_task_index), "global_task": task_instruction}]
    subgoals_rows = [{"subgoal_index": i, "subgoal": name} for i, name in enumerate(subgoal_vocab)]

    episodes_rows = [
        {
            "episode_index": int(r["episode_index"]),
            "tasks": r.get("episode_task_names", [r["task"]]),
            "global_tasks": [r.get("global_task", task_instruction)],
            "subgoals": r.get("episode_subgoal_names", ["<none>"]),
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
    total_chunks = math.ceil(total_episodes / max(int(chunk_size), 1)) if total_episodes > 0 else 0
    total_videos = total_episodes * 3

    info = {
        "codebase_version": "v2.1",
        "robot_type": "x_humanoid",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks_rows),
        "total_global_tasks": len(global_tasks_rows),
        "total_subgoals": len(subgoal_vocab),
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": int(chunk_size),
        "fps": int(fps),
        "use_subgoal_as_task": bool(use_subgoal_as_task),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": feature_spec,
    }

    write_json(dataset_output_dir / "meta" / "info.json", info)
    write_jsonl(dataset_output_dir / "meta" / "tasks.jsonl", tasks_rows)
    write_jsonl(dataset_output_dir / "meta" / "global_tasks.jsonl", global_tasks_rows)
    write_jsonl(dataset_output_dir / "meta" / "subgoals.jsonl", subgoals_rows)
    write_jsonl(dataset_output_dir / "meta" / "episodes.jsonl", episodes_rows)
    write_jsonl(dataset_output_dir / "meta" / "episodes_stats.jsonl", episodes_stats_rows)

    if failures:
        write_jsonl(dataset_output_dir / "meta" / "failed_episodes.jsonl", failures)


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert humanoid HDF5 trajectories to LeRobot v2.1-style datasets. Supports multi-task auto-discovery."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Optional single-task name. If omitted, the script auto-discovers tasks under raw_data_dir.",
    )
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Raw dataset root or single-task directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory")
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
    parser.add_argument(
        "--task_dir_depth",
        type=int,
        default=1,
        help="In auto-discovery mode, how many relative path components under raw_data_dir define one task.",
    )
    parser.add_argument(
        "--annotation_json",
        type=str,
        default=None,
        help="Annotation json file, or a directory containing one or more such json files.",
    )
    parser.add_argument(
        "--use_subgoal_as_task",
        action="store_true",
        help="If enabled, each frame's task_index uses current subgoal_index and tasks.jsonl is written from subgoal vocab. Global task fields are still preserved in parquet/meta.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite each target task dataset directory if it exists")
    parser.add_argument("--skip_failed", action="store_true", help="Continue even if some trajectories fail")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    raw_data_dir = Path(args.raw_data_dir).expanduser().resolve()
    output_root_dir = Path(args.output_dir).expanduser().resolve()

    if not raw_data_dir.exists():
        print(f"[ERROR] raw_data_dir does not exist: {raw_data_dir}", file=sys.stderr)
        return 2

    ensure_dir(output_root_dir)

    ann_index = load_annotation_index(args.annotation_json)
    if ann_index:
        print(f"[INFO] Loaded annotations for {len(ann_index)} episodes from: {args.annotation_json}")
    else:
        print("[INFO] No annotation json provided, subgoal fields will default to <none>")

    task_groups = discover_task_groups(
        raw_data_dir=raw_data_dir,
        task_name=args.task_name,
        task_dir_depth=args.task_dir_depth,
    )
    if not task_groups:
        print(f"[ERROR] No trajectory.hdf5 found under: {raw_data_dir}", file=sys.stderr)
        return 2

    camera_keys = camera_feature_keys(args.camera_key_style)

    task_plans: Dict[str, dict] = {}
    all_jobs: List[dict] = []

    print(f"[INFO] Discovered {len(task_groups)} task(s)")
    for task_idx, (task_key, traj_list) in enumerate(task_groups.items()):
        del task_idx  # unused, kept for stable loop structure

        task_output_name = sanitize_task_output_name(task_key)
        task_instruction = resolve_global_task_instruction(task_key)
        dataset_output_dir = output_root_dir if args.task_name else (output_root_dir / task_output_name)

        if dataset_output_dir.exists() and args.overwrite:
            shutil.rmtree(dataset_output_dir)
        ensure_dir(dataset_output_dir)
        ensure_dir(dataset_output_dir / "meta")

        print(f"[INFO] Task [{task_key}] -> {dataset_output_dir} | {len(traj_list)} trajectories")
        print(f"[INFO] Pre-scanning lengths for task [{task_key}]...")

        valid_traj_list: List[Path] = []
        lengths: List[int] = []
        task_corrupted_records: List[dict] = []
        for traj_path in traj_list:
            ok_len, length, err_info = read_num_steps_safe(traj_path)
            if ok_len:
                valid_traj_list.append(traj_path)
                lengths.append(int(length))
            else:
                task_corrupted_records.append(err_info)
                print(
                    f"[SKIP] task={task_key} | stage=pre_scan | {traj_path} | {err_info['error']}",
                    file=sys.stderr,
                )

        if not valid_traj_list:
            print(f"[WARN] Task [{task_key}] has no valid trajectories after pre-scan, skipping.", file=sys.stderr)
            continue

        global_offsets: List[int] = []
        running = 0
        for length in lengths:
            global_offsets.append(running)
            running += int(length)

        feature_spec = None
        feature_spec_error = None
        for sample_traj in valid_traj_list:
            ok_spec, maybe_spec, err_info = infer_feature_spec_safe(
                sample_traj=sample_traj,
                fps=args.fps,
                camera_keys=camera_keys,
                video_height=args.video_height,
                video_width=args.video_width,
                video_codec=args.video_codec,
            )
            if ok_spec:
                feature_spec = maybe_spec
                break
            task_corrupted_records.append(err_info)
            print(
                f"[SKIP] task={task_key} | stage=infer_feature_spec | {sample_traj} | {err_info['error']}",
                file=sys.stderr,
            )
            feature_spec_error = err_info

        if feature_spec is None:
            print(
                f"[WARN] Task [{task_key}] could not infer feature spec from any valid trajectory, skipping.",
                file=sys.stderr,
            )
            continue

        subgoal_vocab, subgoal_to_idx = collect_subgoal_vocab_for_trajs(valid_traj_list, ann_index)

        task_plans[task_output_name] = {
            "task_key": task_key,
            "task_instruction": task_instruction,
            "global_task_index": 0,
            "dataset_output_dir": dataset_output_dir,
            "feature_spec": feature_spec,
            "results": [],
            "failures": [],
            "corrupted_hdf5s": task_corrupted_records,
            "trajectory_count": len(valid_traj_list),
            "subgoal_vocab": subgoal_vocab,
            "subgoal_to_idx": subgoal_to_idx,
        }

        for episode_index, (traj_path, global_offset) in enumerate(zip(valid_traj_list, global_offsets)):
            ep_key = get_episode_key_from_any_path(traj_path)
            annotation_record = ann_index.get(ep_key)

            all_jobs.append(
                {
                    "trajectory_path": str(traj_path),
                    "output_dir": str(dataset_output_dir),
                    "task_instruction": task_instruction,
                    "task_output_name": task_output_name,
                    "task_index": 0,
                    "global_task_index": 0,
                    "global_task_text": task_instruction,
                    "episode_index": episode_index,
                    "global_index_offset": global_offset,
                    "fps": args.fps,
                    "chunk_size": args.chunk_size,
                    "video_height": args.video_height,
                    "video_width": args.video_width,
                    "video_codec": args.video_codec,
                    "camera_key_style": args.camera_key_style,
                    "annotation_record": annotation_record,
                    "subgoal_to_idx": subgoal_to_idx,
                    "use_subgoal_as_task": args.use_subgoal_as_task,
                }
            )

    print(f"[INFO] Total episodes to convert: {len(all_jobs)}")
    print(f"[INFO] Converting with {args.max_workers} worker(s)...")

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(convert_one_episode, job) for job in all_jobs]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            plan = task_plans[result["task_output_name"]]
            task_key = plan["task_key"]
            if result["ok"]:
                plan["results"].append(result)
                camera_info = result.get("camera_sources", {})
                print(
                    f"[OK] {idx:04d}/{len(all_jobs):04d} | task={task_key} | episode={result['episode_index']:06d} | len={result['length']} | cams=main:{camera_info.get('main','?')},left:{camera_info.get('left','?')},right:{camera_info.get('right','?')} | {result['trajectory_path']}"
                )
            else:
                if result.get("is_probably_corrupted_hdf5", False):
                    plan["corrupted_hdf5s"].append({
                        "trajectory_path": result["trajectory_path"],
                        "stage": "convert_one_episode",
                        "error": result["error"],
                        "traceback": result.get("traceback", ""),
                        "episode_index": result.get("episode_index"),
                        "is_probably_corrupted_hdf5": True,
                    })
                    print(
                        f"[SKIP] {idx:04d}/{len(all_jobs):04d} | task={task_key} | episode={result['episode_index']:06d} | {result['trajectory_path']} | {result['error']}",
                        file=sys.stderr,
                    )
                    continue

                plan["failures"].append(result)
                print(
                    f"[FAIL] {idx:04d}/{len(all_jobs):04d} | task={task_key} | episode={result['episode_index']:06d} | {result['trajectory_path']} | {result['error']}",
                    file=sys.stderr,
                )
                if not args.skip_failed:
                    print(result.get("traceback", ""), file=sys.stderr)
                    return 1

    summary_rows = []
    corrupted_hdf5_rows = []
    for task_output_name, plan in task_plans.items():
        dataset_output_dir = plan["dataset_output_dir"]
        write_task_dataset_meta(
            dataset_output_dir=dataset_output_dir,
            task_instruction=plan["task_instruction"],
            global_task_index=plan.get("global_task_index", 0),
            results=plan["results"],
            feature_spec=plan["feature_spec"],
            chunk_size=args.chunk_size,
            fps=args.fps,
            failures=plan["failures"],
            subgoal_vocab=plan["subgoal_vocab"],
            use_subgoal_as_task=args.use_subgoal_as_task,
        )

        for item in plan.get("corrupted_hdf5s", []):
            corrupted_hdf5_rows.append(
                {
                    "task_key": plan["task_key"],
                    "task_output_name": task_output_name,
                    **item,
                }
            )

        ok_count = len(plan["results"])
        fail_count = len(plan["failures"])
        total_frames = sum(int(r["length"]) for r in plan["results"])
        summary_rows.append(
            {
                "task_key": plan["task_key"],
                "task_output_name": task_output_name,
                "output_dir": str(dataset_output_dir),
                "successful_episodes": ok_count,
                "failed_episodes": fail_count,
                "total_frames": total_frames,
                "total_subgoals": len(plan["subgoal_vocab"]),
            }
        )
        print(
            f"[INFO] Finished task [{plan['task_key']}] | ok={ok_count} | failed={fail_count} | corrupted_skipped={len(plan.get('corrupted_hdf5s', []))} | output={dataset_output_dir}"
        )

    write_json(output_root_dir / "corrupted_hdf5_files.json", corrupted_hdf5_rows)
    print(f"[INFO] Wrote corrupted hdf5 list: {output_root_dir / 'corrupted_hdf5_files.json'}")

    if not args.task_name:
        write_json(output_root_dir / "conversion_summary.json", summary_rows)
        print(f"[INFO] Wrote summary: {output_root_dir / 'conversion_summary.json'}")

    total_ok = sum(row["successful_episodes"] for row in summary_rows)
    total_fail = sum(row["failed_episodes"] for row in summary_rows)
    total_corrupted = len(corrupted_hdf5_rows)
    print(f"[INFO] Done | total_ok={total_ok} | total_failed={total_fail} | total_corrupted_skipped={total_corrupted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
