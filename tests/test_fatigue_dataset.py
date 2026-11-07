"""Model 3 labels and causal features (synthetic)."""
import numpy as np
import pandas as pd

from awc.features.fatigue_dataset import (
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
    speeds = [1, 1, 0.95, 0.85, 0.78]
    assert breakdown_index(speeds, speed_drop=0.20) == 4


def test_risk_horizon():
    risk = risk_labels(6, breakdown_i=4, horizon=2)
    # t=2 warns (4 in 2..4), t=3,4 also; t=5 no (breakdown already at 4, but 4>=5? no)
    assert list(risk) == [0, 0, 1, 1, 1, 0]


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
    # force train split by renaming session into MMFIT train list
    rep_df["session"] = "w01"
    base_df["session"] = "w01"
    raw = build_set_records(rep_df, base_df)
    assert raw[0]["has_baseline"] == 0
    normed, stats = normalize_records(raw, mode="personal")
    # set 1 falls back to global — finite features
    assert np.isfinite(normed[0]["X"]).all()


def test_features_are_causal():
    x = np.zeros((4, 8), dtype=np.float32)
    x[:, 6] = [0, 1 / 3, 2 / 3, 1]  # rep_frac
    x[:, 7] = [0, 1, 2, 3]  # time_since_start
    assert features_are_causal({"X": x})
