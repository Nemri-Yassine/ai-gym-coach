"""Quick-train tests must not touch the real models/ or experiments/ trees."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from awc.config import ROOT

MODELS = ROOT / "models"
EXPERIMENTS = ROOT / "experiments"


def _listing(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {p.name for p in root.iterdir()}


@pytest.fixture
def snapshot_trees():
    before_models = _listing(MODELS)
    before_exps = _listing(EXPERIMENTS)
    yield
    assert _listing(MODELS) == before_models, "models/ changed during quick-train tests"
    assert _listing(EXPERIMENTS) == before_exps, "experiments/ changed during quick-train tests"


def test_quick_trains_leave_real_trees_alone(tmp_path, snapshot_trees):
    env = {
        **os.environ,
        "AWC_MODELS_DIR": str(tmp_path / "models"),
        "AWC_EXPERIMENTS_DIR": str(tmp_path / "experiments"),
    }
    subprocess.check_call(
        [sys.executable, "-m", "awc.training.train_m1", "--arch", "cnn", "--inputs", "pose", "--quick"],
        env=env,
    )
    # In-process M3 quick (same env for models_dir / experiments_dir)
    os.environ["AWC_MODELS_DIR"] = env["AWC_MODELS_DIR"]
    os.environ["AWC_EXPERIMENTS_DIR"] = env["AWC_EXPERIMENTS_DIR"]
    try:
        from awc.features.fatigue_dataset import M3_DIR, build_all
        from awc.training.train_m3 import run_variant

        if not (M3_DIR / "sets_general.npz").is_file():
            pytest.skip("no M3 npz")
        run_variant("B5", quick=True, epochs=1)
    finally:
        os.environ.pop("AWC_MODELS_DIR", None)
        os.environ.pop("AWC_EXPERIMENTS_DIR", None)
