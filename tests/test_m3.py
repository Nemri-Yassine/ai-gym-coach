"""Model 3 forward, rule baseline, lead-time metric, quick train."""
from pathlib import Path

import numpy as np
import torch

from awc.models.m3_fatigue import M3Model, pad_batch, rule_baseline_alert
from awc.training.train_m3 import warning_lead_time


def test_m3_forward_variable_length():
    model = M3Model(in_dim=10, hidden=32)
    xs = [np.random.randn(5, 10).astype(np.float32), np.random.randn(8, 10).astype(np.float32)]
    x, lengths, _ = pad_batch(xs)
    fat, risk = model(x, lengths)
    assert fat.shape == (2, 8)
    assert risk.shape == (2, 8)


def test_rule_baseline():
    speeds = [1.0, 1.0, 0.9, 0.8, 0.7]
    assert rule_baseline_alert(speeds, drop=0.25) == 4


def test_lead_time_metric():
    # breakdown at 5; first alert at 3 → lead 2
    risk = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.9])
    assert warning_lead_time(risk, 5, thresh=0.7) == 2.0
    assert warning_lead_time(risk, 2, thresh=0.7) == 0.0  # alert after breakdown


def test_train_quick(tmp_path, monkeypatch):
    monkeypatch.setenv("AWC_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("AWC_EXPERIMENTS_DIR", str(tmp_path / "experiments"))
    from awc.features.fatigue_dataset import M3_DIR, build_all
    from awc.training.train_m3 import run_variant, speed_loss_scores

    if not (M3_DIR / "sets_general.npz").is_file():
        build_all(use_form=False)
    m = run_variant("B5", quick=True, epochs=1)
    assert "risk_auc" in m
    # continuous B1 score is not a step function
    sc = speed_loss_scores([1.0, 1.0, 0.9, 0.7])
    assert sc[3] > sc[2] > 0
    # quick never writes the real checkpoint name
    assert (tmp_path / "models" / "m3_b5_quick.pt").is_file()
    assert not (tmp_path / "models" / "m3_b5.pt").is_file()
