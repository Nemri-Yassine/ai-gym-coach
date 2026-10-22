"""Sync functions on synthetic rows, plus two checks on real MM-Fit sessions."""
import numpy as np
import pandas as pd
import pytest

from awc.features.sync import attach_labels, hr_to_frame, imu_to_frame, pose_to_frame, sync_session


def test_imu_to_frame_mean():
    arr = np.array([
        [10, 0, 1, 2, 3],
        [10, 10, 3, 4, 5],
        [10, 20, 5, 6, 7],
        [11, 30, 9, 9, 9],
    ], dtype=np.float64)
    out = imu_to_frame(arr, "acc")
    row = out[out["frame"] == 10].iloc[0]
    assert row["acc_x"] == pytest.approx(3)
    assert row["acc_y"] == pytest.approx(4)
    assert row["acc_z"] == pytest.approx(5)
    assert len(out) == 2


def test_hr_interpolation_and_mask():
    arr = np.array([[0, 0, 60], [60, 2000, 120]], dtype=np.float64)
    frames = np.array([0, 30, 60, 220])
    out = hr_to_frame(arr, frames).set_index("frame")
    assert out.loc[30, "hr"] == pytest.approx(90)
    assert out.loc[0, "hr_mask"] == 1
    assert out.loc[60, "hr_mask"] == 1
    assert out.loc[220, "hr_mask"] == 0
    assert np.isnan(out.loc[220, "hr"])


def test_attach_labels():
    df = pd.DataFrame({"frame": np.arange(11)})
    out = attach_labels(df, [(5, 8, 12, "squats")])
    inside = out[out["frame"].between(5, 8)]
    outside = out[~out["frame"].between(5, 8)]
    assert (inside["exercise"] == "squats").all()
    assert (inside["set_id"] == 0).all()
    assert (inside["set_reps"] == 12).all()
    assert (outside["exercise"] == "non_activity").all()
    assert (outside["set_id"] == -1).all()


def test_pose_y_flipped():
    pose = np.zeros((2, 1, 19), dtype=np.float64)
    pose[0, 0, 0] = 7
    pose[1, 0, 1] = 10     # nose, pixel y (down)
    pose[1, 0, 11] = 400   # r_ankle, joint index 10
    out = pose_to_frame(pose)
    assert out.loc[0, "y0"] > out.loc[0, "y10"]
    assert out.loc[0, "y0"] == pytest.approx(-10)
    assert out.loc[0, "frame"] == 7


@pytest.mark.data
def test_sync_w00():
    df = sync_session("w00")
    assert len(df) == 63918
    pose_cols = [f"x{i}" for i in range(18)] + [f"y{i}" for i in range(18)]
    assert df[pose_cols].notna().all().all()
    assert df["hr_mask"].mean() > 0.9
    assert df.loc[df["set_id"] >= 0, "set_id"].nunique() == 30


@pytest.mark.data
def test_session_without_hr():
    df = sync_session("w05")
    assert (df["hr_mask"] == 0).all()
    assert df["hr"].isna().all()
