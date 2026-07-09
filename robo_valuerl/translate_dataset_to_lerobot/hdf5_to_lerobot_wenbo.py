#!/usr/bin/env python3
"""
Convert raw HDF5 trajectories (wenbo/franka format) into LeRobot v2.1-style datasets.

Data layout expected in each trajectory.hdf5:
  puppet/arm_joint_position        float32[T, 8]
  puppet/hand_joint_position       float32[T, 1]
  puppet/end_effector              float32[T, 7]
  observations/rgb_images/camera_left   uint8 JPEG [T]
  observations/rgb_images/camera_right  uint8 JPEG [T]
  observations/rgb_images/camera_wrist  uint8 JPEG [T]

Supports two modes:
1. Single-task mode:
   python script.py --raw_data_dir /path/to/task_dir --output_dir /path/out/task_x --task_name task_x
2. Multi-task auto-discovery mode:
   python script.py --raw_data_dir /path/to/raw_root --output_dir /path/out_root

Output layout (same as LeRobot v2.1):
    <output_dir>/
    ├── task_a/
    │   ├── data/
    │   ├── videos/
    │   └── meta/
    └── conversion_summary.json
"""


import argparse
import json
import math
import os
import shutil
import sys
import traceback
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# -----------------------------
# Constants
# -----------------------------

CAMERA_KEYS = (
    "observation.rgb_images.camera_left",
    "observation.rgb_images.camera_right",
    "observation.rgb_images.camera_wrist",
)

H5_CAMERA_NAMES = ("camera_left", "camera_right", "camera_wrist")


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


def sanitize_task_output_name(task_key: str) -> str:
    parts = [p for p in task_key.replace("\\", "/").split("/") if p]
    if not parts:
        return "root_task"
    return "__".join(parts)


def task_key_to_instruction(task_key: str) -> str:
    return task_key.replace("/", " ").replace("_", " ").strip()


def resolve_task_instruction(task_key: str, meta_task_name: str | None) -> str:
    if meta_task_name is not None and meta_task_name.strip():
        return meta_task_name.strip()
    return task_key_to_instruction(task_key)


# -----------------------------
# Trajectory discovery
# -----------------------------

def discover_trajectories(raw_data_dir: Path) -> List[Path]:
    trajs = sorted(raw_data_dir.rglob("trajectory.hdf5"))
    return [p for p in trajs if p.is_file()]


def discover_task_groups(
    raw_data_dir: Path,
    task_name: str | None,
    task_dir_depth: int,
) -> Dict[str, List[Path]]:
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


# -----------------------------
# HDF5 helpers
# -----------------------------

def read_num_steps(trajectory_path: Path) -> int:
    with h5py.File(trajectory_path, "r") as f:
        return int(f["puppet"]["arm_joint_position"].shape[0])


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


def decode_jpeg_image(raw_bytes: np.ndarray) -> np.ndarray:
    """Decode a JPEG-compressed numpy byte array to an RGB uint8 image."""
    img_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("cv2.imdecode returned None — invalid JPEG data")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


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

def make_video_writer(
    video_path: Path,
    size_wh: Tuple[int, int],
    fps: int,
    codec: str,
) -> cv2.VideoWriter:
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
            return pa.array(
                [np.asarray(v).reshape(-1).tolist() for v in values],
                type=pa.list_(pa.int64()),
            )
        if arr0.dtype.kind == "b":
            return pa.array(
                [np.asarray(v).reshape(-1).tolist() for v in values],
                type=pa.list_(pa.bool_()),
            )
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
    video_height: int,
    video_width: int,
    video_codec: str,
) -> dict:
    with h5py.File(sample_traj, "r") as f:
        features: dict[str, dict[str, Any]] = {}

        def add_vector_feature(key: str, shape: list, dtype: str = "float32") -> None:
            features[key] = {"dtype": dtype, "shape": shape, "names": None}

        for cam_key in CAMERA_KEYS:
            features[cam_key] = {
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

        arm_dim = int(f["puppet"]["arm_joint_position"].shape[1])
        hand_dim = int(f["puppet"]["hand_joint_position"].shape[1])
        ee_dim = int(f["puppet"]["end_effector"].shape[1])

        add_vector_feature("observation.state.arm_joint_position", [arm_dim])
        add_vector_feature("observation.state.hand_joint_position", [hand_dim])
        add_vector_feature("observation.state.end_effector", [ee_dim])
        add_vector_feature("action.arm_joint_position", [arm_dim])
        add_vector_feature("action.hand_joint_position", [hand_dim])
        add_vector_feature("action.end_effector", [ee_dim])

        features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
        features["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
        features["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
        features["index"] = {"dtype": "int64", "shape": [1], "names": None}
        features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}

    return features


def infer_feature_spec_safe(
    sample_traj: Path,
    fps: int,
    video_height: int,
    video_width: int,
    video_codec: str,
) -> tuple[bool, dict | None, dict | None]:
    try:
        spec = infer_feature_spec(
            sample_traj=sample_traj,
            fps=fps,
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
    episode_index = int(job["episode_index"])
    global_index_offset = int(job["global_index_offset"])
    fps = int(job["fps"])
    chunk_size = int(job["chunk_size"])
    video_height = int(job["video_height"])
    video_width = int(job["video_width"])
    video_codec = str(job["video_codec"])

    cam_left_key, cam_right_key, cam_wrist_key = CAMERA_KEYS
    chunk_index = episode_index // chunk_size
    episode_stem = f"episode_{episode_index:06d}"
    chunk_dir = f"chunk-{chunk_index:03d}"

    parquet_path = output_dir / "data" / chunk_dir / f"{episode_stem}.parquet"
    video_left_path = output_dir / "videos" / chunk_dir / cam_left_key / f"{episode_stem}.mp4"
    video_right_path = output_dir / "videos" / chunk_dir / cam_right_key / f"{episode_stem}.mp4"
    video_wrist_path = output_dir / "videos" / chunk_dir / cam_wrist_key / f"{episode_stem}.mp4"

    try:
        with h5py.File(trajectory_path, "r") as f:
            arm_joint = np.array(f["puppet"]["arm_joint_position"], dtype=np.float32)
            hand_joint = np.array(f["puppet"]["hand_joint_position"], dtype=np.float32)
            end_effector = np.array(f["puppet"]["end_effector"], dtype=np.float32)

            rgb_ds = f["observations"]["rgb_images"]
            cam_datasets = {
                "camera_left": rgb_ds["camera_left"],
                "camera_right": rgb_ds["camera_right"],
                "camera_wrist": rgb_ds["camera_wrist"],
            }

            # Clamp to the shortest dataset length
            timesteps = min(
                arm_joint.shape[0],
                len(cam_datasets["camera_left"]),
                len(cam_datasets["camera_right"]),
                len(cam_datasets["camera_wrist"]),
            )
            if timesteps <= 0:
                raise ValueError(f"No usable frames after aligning dataset lengths: {trajectory_path}")

            img_stats = {k: _init_image_stats() for k in CAMERA_KEYS}

            writer_left = make_video_writer(video_left_path, (video_width, video_height), fps, video_codec)
            writer_right = make_video_writer(video_right_path, (video_width, video_height), fps, video_codec)
            writer_wrist = make_video_writer(video_wrist_path, (video_width, video_height), fps, video_codec)

            columns: Dict[str, list] = {
                "observation.state.arm_joint_position": [],
                "observation.state.hand_joint_position": [],
                "observation.state.end_effector": [],
                "action.arm_joint_position": [],
                "action.hand_joint_position": [],
                "action.end_effector": [],
                "timestamp": [],
                "frame_index": [],
                "episode_index": [],
                "index": [],
                "task_index": [],
            }

            try:
                for step in range(timesteps):
                    left_img = decode_jpeg_image(cam_datasets["camera_left"][step])
                    right_img = decode_jpeg_image(cam_datasets["camera_right"][step])
                    wrist_img = decode_jpeg_image(cam_datasets["camera_wrist"][step])

                    left_img = cv2.resize(left_img, (video_width, video_height), interpolation=cv2.INTER_AREA)
                    right_img = cv2.resize(right_img, (video_width, video_height), interpolation=cv2.INTER_AREA)
                    wrist_img = cv2.resize(wrist_img, (video_width, video_height), interpolation=cv2.INTER_AREA)

                    _update_image_stats(img_stats[cam_left_key], left_img)
                    _update_image_stats(img_stats[cam_right_key], right_img)
                    _update_image_stats(img_stats[cam_wrist_key], wrist_img)

                    writer_left.write(cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
                    writer_right.write(cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))
                    writer_wrist.write(cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))

                    columns["observation.state.arm_joint_position"].append(arm_joint[step].tolist())
                    columns["observation.state.hand_joint_position"].append(hand_joint[step].tolist())
                    columns["observation.state.end_effector"].append(end_effector[step].tolist())
                    columns["action.arm_joint_position"].append(arm_joint[step].tolist())
                    columns["action.hand_joint_position"].append(hand_joint[step].tolist())
                    columns["action.end_effector"].append(end_effector[step].tolist())
                    columns["timestamp"].append(float(step / fps))
                    columns["frame_index"].append(int(step))
                    columns["episode_index"].append(int(episode_index))
                    columns["index"].append(int(global_index_offset + step))
                    columns["task_index"].append(int(task_index))
            finally:
                writer_left.release()
                writer_right.release()
                writer_wrist.release()

        write_episode_parquet(parquet_path, columns)

        episode_stats = {
            "observation.state.arm_joint_position": _numeric_stats_from_matrix(
                np.asarray(columns["observation.state.arm_joint_position"], dtype=np.float32)
            ),
            "observation.state.hand_joint_position": _numeric_stats_from_matrix(
                np.asarray(columns["observation.state.hand_joint_position"], dtype=np.float32)
            ),
            "observation.state.end_effector": _numeric_stats_from_matrix(
                np.asarray(columns["observation.state.end_effector"], dtype=np.float32)
            ),
            "action.arm_joint_position": _numeric_stats_from_matrix(
                np.asarray(columns["action.arm_joint_position"], dtype=np.float32)
            ),
            "action.hand_joint_position": _numeric_stats_from_matrix(
                np.asarray(columns["action.hand_joint_position"], dtype=np.float32)
            ),
            "action.end_effector": _numeric_stats_from_matrix(
                np.asarray(columns["action.end_effector"], dtype=np.float32)
            ),
            "timestamp": _numeric_stats_from_matrix(
                np.asarray(columns["timestamp"], dtype=np.float32)
            ),
            "frame_index": _numeric_stats_from_matrix(
                np.asarray(columns["frame_index"], dtype=np.int64)
            ),
            "episode_index": _numeric_stats_from_matrix(
                np.asarray(columns["episode_index"], dtype=np.int64)
            ),
            "index": _numeric_stats_from_matrix(
                np.asarray(columns["index"], dtype=np.int64)
            ),
            "task_index": _numeric_stats_from_matrix(
                np.asarray(columns["task_index"], dtype=np.int64)
            ),
            cam_left_key: _finalize_image_stats(img_stats[cam_left_key]),
            cam_right_key: _finalize_image_stats(img_stats[cam_right_key]),
            cam_wrist_key: _finalize_image_stats(img_stats[cam_wrist_key]),
        }

        return {
            "ok": True,
            "trajectory_path": str(trajectory_path),
            "task_output_name": task_output_name,
            "task": task_instruction,
            "episode_task_names": [task_instruction],
            "episode_index": episode_index,
            "length": timesteps,
            "parquet_path": str(parquet_path.relative_to(output_dir)),
            "video_paths": [
                str(video_left_path.relative_to(output_dir)),
                str(video_right_path.relative_to(output_dir)),
                str(video_wrist_path.relative_to(output_dir)),
            ],
            "episode_stats": episode_stats,
        }

    except Exception as exc:
        tb_text = traceback.format_exc()
        for p in (parquet_path, video_left_path, video_right_path, video_wrist_path):
            safe_unlink(p)
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
# Reindex helpers
# -----------------------------

def _relative_episode_artifact_paths(episode_index: int, chunk_size: int) -> dict[str, Any]:
    chunk_index = episode_index // chunk_size
    episode_stem = f"episode_{episode_index:06d}"
    chunk_dir = f"chunk-{chunk_index:03d}"
    cam_left_key, cam_right_key, cam_wrist_key = CAMERA_KEYS
    return {
        "chunk_index": chunk_index,
        "parquet": Path("data") / chunk_dir / f"{episode_stem}.parquet",
        "videos": [
            Path("videos") / chunk_dir / cam_left_key / f"{episode_stem}.mp4",
            Path("videos") / chunk_dir / cam_right_key / f"{episode_stem}.mp4",
            Path("videos") / chunk_dir / cam_wrist_key / f"{episode_stem}.mp4",
        ],
    }


def _patch_episode_stats_after_reindex(
    episode_stats: dict, episode_index: int, global_index_offset: int, length: int
) -> None:
    episode_stats["episode_index"] = _numeric_stats_from_matrix(
        np.full((length,), int(episode_index), dtype=np.int64)
    )
    episode_stats["index"] = _numeric_stats_from_matrix(
        np.arange(global_index_offset, global_index_offset + length, dtype=np.int64)
    )


def compact_and_reindex_results(
    dataset_output_dir: Path,
    results: List[dict],
    chunk_size: int,
) -> List[dict]:
    """Keep successful episodes contiguous even if some parallel jobs failed."""
    if not results:
        return []

    ordered = sorted(results, key=lambda x: x["episode_index"])
    has_gap = any(
        old_idx != new_idx
        for new_idx, old_idx in enumerate(r["episode_index"] for r in ordered)
    )
    if not has_gap:
        return ordered

    ensure_dir(dataset_output_dir / ".reindex_tmp")
    staging_root = Path(
        tempfile.mkdtemp(
            prefix="episode_reindex_", dir=str(dataset_output_dir / ".reindex_tmp")
        )
    )

    staged_rows: List[dict] = []
    try:
        for row in ordered:
            old_paths = _relative_episode_artifact_paths(
                episode_index=int(row["episode_index"]),
                chunk_size=int(chunk_size),
            )
            old_parquet_abs = dataset_output_dir / old_paths["parquet"]
            old_video_abs_list = [dataset_output_dir / p for p in old_paths["videos"]]

            stage_name = f"episode_{int(row['episode_index']):06d}"
            staged_parquet = staging_root / f"{stage_name}.parquet"
            staged_videos = [
                staging_root / f"{stage_name}.video{idx}.mp4"
                for idx, _ in enumerate(old_video_abs_list)
            ]

            ensure_dir(staged_parquet.parent)
            shutil.move(str(old_parquet_abs), str(staged_parquet))
            for src, dst in zip(old_video_abs_list, staged_videos):
                ensure_dir(dst.parent)
                shutil.move(str(src), str(dst))

            staged_rows.append(
                {
                    "result": row,
                    "staged_parquet": staged_parquet,
                    "staged_videos": staged_videos,
                }
            )

        global_index_offset = 0
        for new_episode_index, item in enumerate(staged_rows):
            row = item["result"]
            new_paths = _relative_episode_artifact_paths(
                episode_index=new_episode_index,
                chunk_size=int(chunk_size),
            )
            new_parquet_abs = dataset_output_dir / new_paths["parquet"]
            new_video_abs_list = [dataset_output_dir / p for p in new_paths["videos"]]

            table = pq.read_table(item["staged_parquet"])
            columns = {name: table[name].to_pylist() for name in table.column_names}
            length = int(row["length"])

            if len(columns.get("frame_index", [])) != length:
                raise ValueError(
                    f"Length mismatch while reindexing {item['staged_parquet']}: "
                    f"expected {length}, got {len(columns.get('frame_index', []))}"
                )

            columns["episode_index"] = [int(new_episode_index)] * length
            columns["index"] = [int(global_index_offset + i) for i in range(length)]

            write_episode_parquet(new_parquet_abs, columns)
            for staged_video, final_video in zip(item["staged_videos"], new_video_abs_list):
                ensure_dir(final_video.parent)
                shutil.move(str(staged_video), str(final_video))

            row["episode_index"] = int(new_episode_index)
            row["parquet_path"] = str(new_paths["parquet"])
            row["video_paths"] = [str(p) for p in new_paths["videos"]]
            _patch_episode_stats_after_reindex(
                episode_stats=row["episode_stats"],
                episode_index=int(new_episode_index),
                global_index_offset=int(global_index_offset),
                length=length,
            )
            global_index_offset += length

        return ordered
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


# -----------------------------
# Metadata writing
# -----------------------------

def write_task_dataset_meta(
    dataset_output_dir: Path,
    task_instruction: str,
    results: List[dict],
    feature_spec: dict,
    chunk_size: int,
    fps: int,
    failures: List[dict],
) -> None:
    results = sorted(results, key=lambda x: x["episode_index"])

    tasks_rows = [{"task_index": 0, "task": task_instruction}]
    episodes_rows = [
        {
            "episode_index": int(r["episode_index"]),
            "tasks": r.get("episode_task_names", [r["task"]]),
            "length": int(r["length"]),
        }
        for r in results
    ]
    episodes_stats_rows = [
        {"episode_index": int(r["episode_index"]), "stats": r["episode_stats"]}
        for r in results
    ]

    total_episodes = len(results)
    total_frames = sum(int(r["length"]) for r in results)
    total_chunks = (
        math.ceil(total_episodes / max(int(chunk_size), 1)) if total_episodes > 0 else 0
    )
    total_videos = total_episodes * len(CAMERA_KEYS)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka_emika_singleArm-gripper",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks_rows),
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": int(chunk_size),
        "fps": int(fps),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": feature_spec,
    }

    write_json(dataset_output_dir / "meta" / "info.json", info)
    write_jsonl(dataset_output_dir / "meta" / "tasks.jsonl", tasks_rows)
    write_jsonl(dataset_output_dir / "meta" / "episodes.jsonl", episodes_rows)
    write_jsonl(dataset_output_dir / "meta" / "episodes_stats.jsonl", episodes_stats_rows)

    if failures:
        write_jsonl(dataset_output_dir / "meta" / "failed_episodes.jsonl", failures)


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert wenbo/franka HDF5 trajectories to LeRobot v2.1-style datasets."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Optional single-task name. If omitted, auto-discovers tasks under raw_data_dir.",
    )
    parser.add_argument(
        "--meta_task_name",
        type=str,
        default=None,
        help="Optional explicit task instruction written into meta. Overrides folder-derived name.",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        required=True,
        help="Raw dataset root or single-task directory.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Episodes per chunk directory.")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=max(os.cpu_count() or 1, 1),
        help="Parallel worker count.",
    )
    parser.add_argument("--video_height", type=int, default=480, help="Output video height.")
    parser.add_argument("--video_width", type=int, default=640, help="Output video width.")
    parser.add_argument(
        "--video_codec",
        type=str,
        default="mp4v",
        help="Preferred fourcc codec, e.g. mp4v or avc1.",
    )
    parser.add_argument(
        "--task_dir_depth",
        type=int,
        default=1,
        help="In auto-discovery mode, how many relative path components define one task.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite target task dataset directory if it exists.",
    )
    parser.add_argument(
        "--skip_failed",
        action="store_true",
        help="Continue even if some trajectories fail.",
    )
    parser.add_argument(
        "--is_single_task",
        action="store_true",
        help="Output directly to output_dir (no subtask subdirectory).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    raw_data_dir = Path(args.raw_data_dir).expanduser().resolve()
    output_root_dir = Path(args.output_dir).expanduser().resolve()

    if not raw_data_dir.exists():
        print(f"[ERROR] raw_data_dir does not exist: {raw_data_dir}", file=sys.stderr)
        return 2

    ensure_dir(output_root_dir)

    task_groups = discover_task_groups(
        raw_data_dir=raw_data_dir,
        task_name=args.task_name,
        task_dir_depth=args.task_dir_depth,
    )
    if not task_groups:
        print(f"[ERROR] No trajectory.hdf5 found under: {raw_data_dir}", file=sys.stderr)
        return 2

    task_plans: Dict[str, dict] = {}
    all_jobs: List[dict] = []

    print(f"[INFO] Discovered {len(task_groups)} task(s)")
    for task_key, traj_list in task_groups.items():
        task_output_name = sanitize_task_output_name(task_key)
        task_instruction = resolve_task_instruction(task_key, args.meta_task_name)
        dataset_output_dir = (
            output_root_dir
            if (args.task_name or args.is_single_task)
            else (output_root_dir / task_output_name)
        )

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
            print(
                f"[WARN] Task [{task_key}] has no valid trajectories after pre-scan, skipping.",
                file=sys.stderr,
            )
            continue

        global_offsets: List[int] = []
        running = 0
        for length in lengths:
            global_offsets.append(running)
            running += int(length)

        feature_spec = None
        for sample_traj in valid_traj_list:
            ok_spec, maybe_spec, err_info = infer_feature_spec_safe(
                sample_traj=sample_traj,
                fps=args.fps,
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

        if feature_spec is None:
            print(
                f"[WARN] Task [{task_key}] could not infer feature spec from any valid trajectory, skipping.",
                file=sys.stderr,
            )
            continue

        task_plans[task_output_name] = {
            "task_key": task_key,
            "task_instruction": task_instruction,
            "dataset_output_dir": dataset_output_dir,
            "feature_spec": feature_spec,
            "results": [],
            "failures": [],
            "corrupted_hdf5s": task_corrupted_records,
            "trajectory_count": len(valid_traj_list),
        }

        for episode_index, (traj_path, global_offset) in enumerate(
            zip(valid_traj_list, global_offsets)
        ):
            all_jobs.append(
                {
                    "trajectory_path": str(traj_path),
                    "output_dir": str(dataset_output_dir),
                    "task_instruction": task_instruction,
                    "task_output_name": task_output_name,
                    "task_index": 0,
                    "episode_index": episode_index,
                    "global_index_offset": global_offset,
                    "fps": args.fps,
                    "chunk_size": args.chunk_size,
                    "video_height": args.video_height,
                    "video_width": args.video_width,
                    "video_codec": args.video_codec,
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
                print(
                    f"[OK]   {idx:04d}/{len(all_jobs):04d} | task={task_key} "
                    f"| episode={result['episode_index']:06d} | len={result['length']} "
                    f"| {result['trajectory_path']}"
                )
            else:
                if result.get("is_probably_corrupted_hdf5", False):
                    plan["corrupted_hdf5s"].append(
                        {
                            "trajectory_path": result["trajectory_path"],
                            "stage": "convert_one_episode",
                            "error": result["error"],
                            "traceback": result.get("traceback", ""),
                            "episode_index": result.get("episode_index"),
                            "is_probably_corrupted_hdf5": True,
                        }
                    )
                    print(
                        f"[SKIP] {idx:04d}/{len(all_jobs):04d} | task={task_key} "
                        f"| episode={result['episode_index']:06d} | {result['trajectory_path']} "
                        f"| {result['error']}",
                        file=sys.stderr,
                    )
                    continue

                plan["failures"].append(result)
                print(
                    f"[FAIL] {idx:04d}/{len(all_jobs):04d} | task={task_key} "
                    f"| episode={result['episode_index']:06d} | {result['trajectory_path']} "
                    f"| {result['error']}",
                    file=sys.stderr,
                )
                print(result.get("traceback", ""), file=sys.stderr)
                if not args.skip_failed:
                    return 1

    summary_rows = []
    corrupted_hdf5_rows = []
    for task_output_name, plan in task_plans.items():
        dataset_output_dir = plan["dataset_output_dir"]
        plan["results"] = compact_and_reindex_results(
            dataset_output_dir=dataset_output_dir,
            results=plan["results"],
            chunk_size=args.chunk_size,
        )
        write_task_dataset_meta(
            dataset_output_dir=dataset_output_dir,
            task_instruction=plan["task_instruction"],
            results=plan["results"],
            feature_spec=plan["feature_spec"],
            chunk_size=args.chunk_size,
            fps=args.fps,
            failures=plan["failures"],
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
            }
        )
        print(
            f"[INFO] Finished task [{plan['task_key']}] | ok={ok_count} | failed={fail_count} "
            f"| corrupted_skipped={len(plan.get('corrupted_hdf5s', []))} | output={dataset_output_dir}"
        )

    write_json(output_root_dir / "corrupted_hdf5_files.json", corrupted_hdf5_rows)
    print(f"[INFO] Wrote corrupted hdf5 list: {output_root_dir / 'corrupted_hdf5_files.json'}")

    if not args.task_name and not args.is_single_task:
        write_json(output_root_dir / "conversion_summary.json", summary_rows)
        print(f"[INFO] Wrote summary: {output_root_dir / 'conversion_summary.json'}")

    total_ok = sum(row["successful_episodes"] for row in summary_rows)
    total_fail = sum(row["failed_episodes"] for row in summary_rows)
    total_corrupted = len(corrupted_hdf5_rows)
    print(
        f"[INFO] Done | total_ok={total_ok} | total_failed={total_fail} "
        f"| total_corrupted_skipped={total_corrupted}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
