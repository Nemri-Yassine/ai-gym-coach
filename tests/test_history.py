"""Within-session baselines and rep-table schema (synthetic; no dataset needed)."""
import numpy as np
import pandas as pd
import pytest

from awc.bigdata.history import (
    REP_COLUMNS,
    add_within_session_baselines,
    build_rep_rows_for_session,
)
from awc.config import PROCESSED
from awc.features.reps import rep_boundaries


def test_rep_boundaries_sine():
    t = np.linspace(0, 10 * 2 * np.pi, 300, endpoint=False)
    pose = np.zeros((300, 18, 2), dtype=np.float64)
    pose[:, 8, 1] = np.sin(t)
    pose[:, 1, 1] = 1.0
    bounds = rep_boundaries(pose)
    assert len(bounds) == 10
    assert all(a < b for a, b in bounds)


def test_baseline_uses_only_past():
    """Set 3 baseline = mean of sets 1–2; set 1 has NaN and has_baseline = 0."""
    rows = []
    for set_idx in (1, 2, 3):
        for rep in range(3):
            rows.append({
                "source": "mmfit", "user": "wX", "session": "wX", "session_date": "wX",
                "exercise": "squats", "set_id": set_idx - 1, "set_idx_in_exercise": set_idx,
                "rep_idx": rep, "n_reps_in_set": 3,
                "duration_s": 1.0 * set_idx, "rom": 2.0 * set_idx, "mean_speed": 3.0 * set_idx,
                "form_score": np.nan, "hr_mean": 100.0, "hr_end": 100.0 + set_idx,
                "hr_rise": float(set_idx), "hr_mask": 1,
            })
    rep_df = pd.DataFrame(rows)
    bases = add_within_session_baselines(rep_df)
    b1 = bases[bases["set_idx_in_exercise"] == 1].iloc[0]
    b3 = bases[bases["set_idx_in_exercise"] == 3].iloc[0]
    assert b1["has_baseline"] == 0
    assert pd.isna(b1["base_duration_mean"])
    assert b3["has_baseline"] == 1
    # first-3 mean duration for set1=1, set2=2 → mean 1.5
    assert abs(b3["base_duration_mean"] - 1.5) < 1e-9
    assert abs(b3["base_rom_mean"] - 3.0) < 1e-9
    assert abs(b3["base_speed_mean"] - 4.5) < 1e-9
    assert abs(b3["base_hr_rise_mean"] - 1.5) < 1e-9


def test_rep_table_columns():
    df = pd.DataFrame(columns=REP_COLUMNS)
    assert list(df.columns) == REP_COLUMNS


@pytest.mark.data
def test_rep_table_row_count():
    path = PROCESSED / "rep_table.parquet"
    if not path.is_file():
        pytest.skip("rep_table.parquet not built yet")
    df = pd.read_parquet(path)
    assert len(df) > 5000
