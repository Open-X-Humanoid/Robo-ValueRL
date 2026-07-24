"""Centralized, machine-agnostic path configuration for reproduction.

Reproducers do NOT need to edit any config file. Instead, point these two
environment variables at wherever you placed the datasets and checkpoints:

    export RVRL_DATA_ROOT=/your/datasets      # where the LeRobot datasets live
    export RVRL_CKPT_ROOT=/your/checkpoints   # where model weights / checkpoints live

The dataset / checkpoint sub-directory structure used by the released configs
is preserved relative to these roots, so the sub-paths themselves document the
layout that each config expects (e.g. ``rl_block_x_humanoid_data/...``).

`norm_stats` defaults to the copy shipped inside this repository; override it
with ``RVRL_NORM_STATS`` if you recomputed your own statistics.
"""

import os
from pathlib import Path

# Repository root = corl_release_code/  (this file is at
# openpi/src/openpi/training/paths.py, so go up 4 levels).
REPO_ROOT = Path(__file__).resolve().parents[4]

# Root directories. Defaults are repo-relative so a fresh clone at least runs
# without any environment setup; override via env vars for real data locations.
DATA_ROOT = Path(os.environ.get("RVRL_DATA_ROOT", REPO_ROOT / "data"))
CKPT_ROOT = Path(os.environ.get("RVRL_CKPT_ROOT", REPO_ROOT / "checkpoints"))

# Normalization stats: ship-in-repo default, overridable.
NORM_STATS_PATH = os.environ.get(
    "RVRL_NORM_STATS",
    str(REPO_ROOT / "robo_valuerl" / "humanoid_delta_fast_stats.json"),
)


def data(*parts: str) -> str:
    """Build an absolute path under the dataset root."""
    return str(DATA_ROOT.joinpath(*parts))


def ckpt(*parts: str) -> str:
    """Build an absolute path under the checkpoint root."""
    return str(CKPT_ROOT.joinpath(*parts))
