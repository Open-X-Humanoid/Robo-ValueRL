"""
Chunked batch conversion of jidi HDF5 sessions to LeRobot format.

Each session folder is converted to its own independent LeRobot dataset.
Conversion logic is identical to translate_x_humanoid_three_camera_with_quality_for_jidi_data.py.

Data layout (input):
  {hdf5_classified_dir}/
    excellent/  (24 sessions)
    ordinary/   (17 sessions)
    unlabeled/  (35 sessions)
      └── {session_name}/
            └── success_episodes/
                  └── {timestamp}/
                        └── data/trajectory.hdf5

Output per session:
  {output_dir}/{category}/{session_name}/   ← one LeRobot dataset each

Sessions are sorted within each category, then ordered: excellent → ordinary → unlabeled.
They are divided into --num_chunks equal slices; --chunk_index (1-based) selects one slice.

Examples:
  # Show chunk assignments (no conversion)
  python translate_x_humanoid_jidi_chunked.py --list_chunks

  # Convert chunk 1
  python translate_x_humanoid_jidi_chunked.py \\
    --chunk_index 1 \\
    --output_dir /path/to/output

  # Custom split (6 chunks)
  python translate_x_humanoid_jidi_chunked.py \\
    --num_chunks 6 --chunk_index 3 \\
    --output_dir /path/to/output
"""

import argparse
import os
import time

from features import X_HUMANOID_REAL_WORLD_FEATURES
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from translate_x_humanoid_three_camera_with_quality_for_jidi_data import load_trajectory

DATA_CATEGORIES = ["excellent", "ordinary", "unlabeled"]


# ---------------------------------------------------------------------------
# Session / chunk helpers
# ---------------------------------------------------------------------------

def collect_sessions(hdf5_classified_dir: str) -> list[tuple[str, str, str]]:
    """Return sorted list of (category, session_name, session_path) across all categories."""
    sessions = []
    for category in DATA_CATEGORIES:
        cat_path = os.path.join(hdf5_classified_dir, category)
        if not os.path.isdir(cat_path):
            continue
        for session_name in sorted(os.listdir(cat_path)):
            session_path = os.path.join(cat_path, session_name)
            if os.path.isdir(session_path):
                sessions.append((category, session_name, session_path))
    return sessions


def split_chunks(sessions: list, num_chunks: int) -> list[list]:
    """Split session list into num_chunks chunks (earlier chunks may be 1 larger)."""
    total = len(sessions)
    base, remainder = divmod(total, num_chunks)
    chunks, start = [], 0
    for i in range(num_chunks):
        size = base + (1 if i < remainder else 0)
        chunks.append(sessions[start: start + size])
        start += size
    return chunks


def find_trajectories(session_path: str) -> list[str]:
    """Find all trajectory.hdf5 files under session_path/success_episodes/."""
    traj_paths = []
    episodes_dir = os.path.join(session_path, "success_episodes")
    if not os.path.isdir(episodes_dir):
        return traj_paths
    for ts in sorted(os.listdir(episodes_dir)):
        traj = os.path.join(episodes_dir, ts, "data", "trajectory.hdf5")
        if os.path.isfile(traj):
            traj_paths.append(traj)
    return traj_paths


# ---------------------------------------------------------------------------
# Per-session conversion
# ---------------------------------------------------------------------------

def convert_session(session_path: str, output_dir: str, repo_id: str,
                    task_instruction: str, fps: int) -> None:
    """Convert all trajectories in one session to an independent LeRobot dataset."""
    traj_list = find_trajectories(session_path)
    if not traj_list:
        print(f"  [WARN] No trajectory files found, skipping: {session_path}")
        return

    print(f"  {len(traj_list)} trajectories → {output_dir}")

    dataset = LeRobotDataset.create(
        features=X_HUMANOID_REAL_WORLD_FEATURES,
        root=output_dir,
        repo_id=repo_id,
        fps=fps,
        robot_type="x_humanoid",
    )

    for traj_index, traj_path in enumerate(traj_list):
        t0 = time.time()
        trajectory = load_trajectory(traj_path, task_instruction)
        if trajectory is None:
            continue
        for step_data in trajectory:
            dataset.add_frame(step_data)
        dataset.save_episode()
        print(f"    [{traj_index + 1}/{len(traj_list)}] {time.time() - t0:.1f}s  {traj_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Chunked batch LeRobot conversion — one dataset per session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hdf5_classified_dir",
        type=str,
        default="/mnt/dataset/toago6/workspace/code/rl_block_x_humanoid_data/jidi_robot_data/hdf5_classified",
        help="Root dir containing excellent/ordinary/unlabeled subdirs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base output directory; each session is saved under {output_dir}/{category}/{session_name}",
    )
    parser.add_argument("--num_chunks", type=int, default=24, help="Total number of chunks")
    parser.add_argument(
        "--chunk_index",
        type=int,
        default=None,
        help="1-based index of the chunk to convert",
    )
    parser.add_argument(
        "--list_chunks",
        action="store_true",
        help="Print chunk assignments and exit without converting",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="separate_the_blocks_and_sort_by_color_into_different_plates",
    )
    parser.add_argument("--repo_id_prefix", type=str, default="x_humanoid",
                        help="Repo ID prefix; full ID becomes {prefix}/{session_name}")
    parser.add_argument("--fps", type=int, default=30)

    args = parser.parse_args()

    all_sessions = collect_sessions(args.hdf5_classified_dir)
    if not all_sessions:
        print(f"No sessions found under: {args.hdf5_classified_dir}")
        return

    chunks = split_chunks(all_sessions, args.num_chunks)

    # --list_chunks: show mapping and exit
    if args.list_chunks:
        print(f"Total sessions: {len(all_sessions)}  |  num_chunks: {args.num_chunks}\n")
        for idx, chunk in enumerate(chunks, start=1):
            cats = {}
            for cat, name, _ in chunk:
                cats.setdefault(cat, []).append(name)
            summary = ", ".join(f"{len(v)} {k}" for k, v in cats.items())
            print(f"Chunk {idx:>2}/{args.num_chunks}  ({len(chunk)} sessions)  [{summary}]")
            for cat, name, _ in chunk:
                print(f"    [{cat:10s}]  {name}")
        return

    # Validate args for actual conversion
    if args.chunk_index is None:
        parser.error("--chunk_index is required when not using --list_chunks")
    if not (1 <= args.chunk_index <= args.num_chunks):
        parser.error(f"--chunk_index must be between 1 and {args.num_chunks}")
    if args.output_dir is None:
        parser.error("--output_dir is required when converting")

    selected = chunks[args.chunk_index - 1]
    task_instruction = " ".join(args.task_name.split("_"))

    print(f"Chunk {args.chunk_index}/{args.num_chunks}: {len(selected)} sessions\n")

    for session_idx, (category, session_name, session_path) in enumerate(selected, start=1):
        session_output = os.path.join(args.output_dir, category, session_name)
        repo_id = f"{args.repo_id_prefix}/{session_name}"
        print(f"[{session_idx}/{len(selected)}] [{category}] {session_name}")
        convert_session(session_path, session_output, repo_id, task_instruction, args.fps)
        print()

    print(f"Done. All sessions saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
