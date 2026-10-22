"""Join MM-Fit streams on the shared 30 fps frame index.

Pose is the clock. Watch IMU (~100 Hz) is averaged onto each frame.
Heart rate (~1 Hz) is linearly interpolated, with a mask for gaps longer than 5 s.
"""
import numpy as np
import pandas as pd

from awc.config import MMFIT_SPLIT
from awc.data import mmfit

POSE_X = [f"x{i}" for i in range(18)]
POSE_Y = [f"y{i}" for i in range(18)]
IMU_COLS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
SYNC_COLUMNS = (
    ["frame"] + POSE_X + POSE_Y + IMU_COLS + ["hr", "hr_mask", "exercise", "set_id", "set_reps", "session", "split"]
)
HR_GAP_FRAMES = 150  # ±5 s at 30 fps


def session_split(session_id):
    """Split name from MMFIT_SPLIT, or 'unused' when the session is in none of them."""
    for name, ids in MMFIT_SPLIT.items():
        if session_id in ids:
            return name
    return "unused"


def pose_to_frame(pose_2d) -> pd.DataFrame:
    """(2, frames, 19) → frame, x0..x17, y0..y17. Pixel y points down; we store y up."""
    pose = np.asarray(pose_2d)
    if pose.ndim != 3 or pose.shape[0] != 2 or pose.shape[2] < 19:
        raise ValueError(f"pose_2d must be (2, frames, 19+), got {pose.shape}")
    frames = np.rint(pose[0, :, 0]).astype(np.int64)
    xs = np.asarray(pose[0, :, 1:19], dtype=np.float64)
    ys = -np.asarray(pose[1, :, 1:19], dtype=np.float64)
    data = {"frame": frames}
    for i in range(18):
        data[f"x{i}"] = xs[:, i]
        data[f"y{i}"] = ys[:, i]
    return pd.DataFrame(data)


def imu_to_frame(arr, prefix) -> pd.DataFrame:
    """IMU rows [frame, ts_ms, x, y, z] → one row per frame, mean of the samples on that frame."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return pd.DataFrame(columns=["frame", f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"])
    df = pd.DataFrame({
        "frame": np.rint(arr[:, 0]).astype(np.int64),
        f"{prefix}_x": arr[:, 2],
        f"{prefix}_y": arr[:, 3],
        f"{prefix}_z": arr[:, 4],
    })
    return df.groupby("frame", as_index=False).mean()


def hr_to_frame(arr, frames) -> pd.DataFrame:
    """Linear bpm interpolation. hr_mask is 1 only when a real sample lies within ±150 frames."""
    frames = np.asarray(frames)
    n = len(frames)
    hr = np.full(n, np.nan, dtype=np.float64)
    mask = np.zeros(n, dtype=np.int32)
    if arr is None:
        return pd.DataFrame({"frame": frames, "hr": hr, "hr_mask": mask})
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return pd.DataFrame({"frame": frames, "hr": hr, "hr_mask": mask})
    order = np.argsort(arr[:, 0])
    samples = arr[order, 0]
    bpm = arr[order, 2]
    uniq, idx = np.unique(samples, return_index=True)
    samples, bpm = uniq.astype(np.float64), bpm[idx].astype(np.float64)
    hr = np.interp(frames.astype(np.float64), samples, bpm)
    pos = np.searchsorted(samples, frames)
    right = np.clip(pos, 0, len(samples) - 1)
    left = np.clip(pos - 1, 0, len(samples) - 1)
    dist = np.minimum(np.abs(frames - samples[left]), np.abs(frames - samples[right]))
    mask = (dist <= HR_GAP_FRAMES).astype(np.int32)
    hr = hr.astype(np.float64)
    hr[mask == 0] = np.nan
    return pd.DataFrame({"frame": frames, "hr": hr, "hr_mask": mask})


def attach_labels(df, labels) -> pd.DataFrame:
    """Add exercise, set_id and set_reps. Frames outside every set are non_activity, set_id -1."""
    out = df.copy()
    exercise = np.full(len(out), "non_activity", dtype=object)
    set_id = np.full(len(out), -1, dtype=np.int32)
    set_reps = np.zeros(len(out), dtype=np.int32)
    frames = out["frame"].to_numpy()
    for i, (start, end, reps, name) in enumerate(labels):
        inside = (frames >= start) & (frames <= end)
        exercise[inside] = name
        set_id[inside] = i
        set_reps[inside] = reps
    out["exercise"] = exercise
    out["set_id"] = set_id
    out["set_reps"] = set_reps
    return out


def _as_copied(stream):
    if stream is None:
        return None
    return np.array(stream)


def sync_session(session_id) -> pd.DataFrame:
    """One session at the pose clock: pose + watch IMU + heart rate + set label + split."""
    pose = _as_copied(mmfit.load_stream(session_id, "pose_2d", mmap=False))
    if pose is None:
        raise FileNotFoundError(f"{session_id} has no pose_2d")
    df = pose_to_frame(pose)
    for stream, prefix in (("sw_l_acc", "acc"), ("sw_l_gyr", "gyr")):
        raw = _as_copied(mmfit.load_stream(session_id, stream, mmap=False))
        if raw is None:
            for axis in "xyz":
                df[f"{prefix}_{axis}"] = np.nan
            continue
        df = df.merge(imu_to_frame(raw, prefix), on="frame", how="left")
    hr = hr_to_frame(_as_copied(mmfit.load_stream(session_id, "sw_l_hr", mmap=False)), df["frame"].to_numpy())
    df["hr"] = hr["hr"].to_numpy()
    df["hr_mask"] = hr["hr_mask"].to_numpy()
    df = attach_labels(df, mmfit.load_labels(session_id))
    df["session"] = session_id
    df["split"] = session_split(session_id)
    df["frame"] = df["frame"].astype(np.int64)
    df["hr_mask"] = df["hr_mask"].astype(np.int32)
    df["set_id"] = df["set_id"].astype(np.int32)
    df["set_reps"] = df["set_reps"].astype(np.int32)
    for col in POSE_X + POSE_Y + IMU_COLS + ["hr"]:
        df[col] = df[col].astype(np.float64)
    return df.loc[:, SYNC_COLUMNS]
