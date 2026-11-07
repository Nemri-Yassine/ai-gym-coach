"""Model 3 labels and causal features (synthetic)."""
import json

import numpy as np
import pandas as pd
import pytest

from awc.config import PROCESSED
from awc.features.fatigue_dataset import (
    M3_DIR,
    breakdown_index,
    build_set_records,
    features_are_causal,
    fatigue_labels,
    normalize_records,
    risk_labels,
)


def test_fatigue_label_range():
    y = fatigue_labels(5)
    assert y[0] == 0.0 and y[-1] == 1.0
    assert np.all(np.diff(y) > 0)


def test_breakdown_detection():
    # speed-only default: 0.78 is first ≤ 0.8 * mean(1,1)
    speeds = [1, 1, 0.95, 0.85, 0.78, 0.7]
    assert breakdown_index(speeds, speed_drop=0.20) == 4
    assert breakdown_index(speeds, speed_drop=0.20, use_form=False) == 4
    # form alone must NOT fire when use_form=False
    form = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    assert breakdown_index([1, 1, 1, 1, 1, 1], form, use_form=False) is None
    assert breakdown_index([1, 1, 1, 1, 1, 1], form, use_form=True, form_thresh=0.5) == 2


def test_risk_horizon():
    # breakdown at 4, horizon 2 → risk from index 2 through end
    risk = risk_labels(6, breakdown_i=4, horizon=2)
    assert list(risk) == [0, 0, 1, 1, 1, 1]


def test_personal_norm_fallback():
    rows = []
    for set_idx in (1, 2):
        for rep in range(3):
            rows.append({
                "session": "wX", "exercise": "squats", "set_id": set_idx - 1,
                "set_idx_in_exercise": set_idx, "rep_idx": rep,
                "duration_s": 1.0, "rom": 1.0, "mean_speed": 1.0,
                "form_score": np.nan, "hr_end": 100.0, "hr_rise": 1.0, "hr_mask": 1,
            })
    rep_df = pd.DataFrame(rows)
    base_df = pd.DataFrame([
        {"session": "wX", "exercise": "squats", "set_id": 0, "has_baseline": 0,
         "base_duration_mean": np.nan, "base_duration_std": np.nan,
         "base_rom_mean": np.nan, "base_rom_std": np.nan,
         "base_speed_mean": np.nan, "base_speed_std": np.nan,
         "base_hr_rise_mean": np.nan, "base_hr_rise_std": np.nan},
        {"session": "wX", "exercise": "squats", "set_id": 1, "has_baseline": 1,
         "base_duration_mean": 1.0, "base_duration_std": 0.1,
         "base_rom_mean": 1.0, "base_rom_std": 0.1,
         "base_speed_mean": 1.0, "base_speed_std": 0.1,
         "base_hr_rise_mean": 1.0, "base_hr_rise_std": 0.1},
    ])
    rep_df["session"] = "w01"
    base_df["session"] = "w01"
    raw = build_set_records(rep_df, base_df)
    assert raw[0]["has_baseline"] == 0
    normed, stats = normalize_records(raw, mode="personal")
    assert np.isfinite(normed[0]["X"]).all()


def test_features_are_causal():
    x = np.zeros((4, 8), dtype=np.float32)
    x[:, 6] = [0, 1 / 3, 2 / 3, 1]
    x[:, 7] = [0, 1, 2, 3]
    assert features_are_causal({"X": x})


@pytest.mark.data
def test_breakdown_rates_not_extreme():
    """Flag exercises with breakdown rate > 90% or < 5%; those must be named in the README."""
    from pathlib import Path

    from awc.config import ROOT

    path = M3_DIR / "breakdown_stats.json"
    if not path.is_file():
        pytest.skip("breakdown_stats.json missing; run build_all first")
    rows = json.loads(path.read_text(encoding="utf-8"))
    bad = [r for r in rows if r["n_sets"] >= 10 and (r["breakdown_rate"] > 0.90 or r["breakdown_rate"] < 0.05)]
    if not bad:
        return
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    missing = [r["exercise"] for r in bad if r["exercise"].replace("_", " ") not in readme and r["exercise"] not in readme]
    assert not missing, (
        f"extreme breakdown rates {bad} must be named in README.md (missing: {missing})"
    )
