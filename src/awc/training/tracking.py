"""Reproducible experiment folders: config.json, metrics.json, figures."""
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from awc.config import EXPERIMENTS, ROOT


def seed_everything(seed=42):
    """Fix python, numpy and torch RNGs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit():
    """Current HEAD hash, or None when the repo has no commit yet."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return out.strip() or None


def new_run(name, config) -> Path:
    """Create experiments/<YYYYMMDD>_<name>/ and write config.json (includes git_commit)."""
    run_dir = EXPERIMENTS / f"{datetime.now():%Y%m%d}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(config)
    payload["git_commit"] = git_commit()
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_dir


def save_metrics(run_dir, metrics: dict):
    """Write metrics.json into an existing run folder."""
    path = Path(run_dir) / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path
