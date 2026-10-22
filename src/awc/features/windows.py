"""Cut synced sessions into 3 s windows (90 frames, hop 45) for Model 1."""
import json

import numpy as np
import pandas as pd

from awc.config import INTERIM, MMFIT_CLASSES, PROCESSED, VIDEO_FPS, WINDOW_SECONDS
from awc.features.skeleton import normalize
from awc.features.sync import IMU_COLS, POSE_X, POSE_Y

WINDOW = int(WINDOW_SECONDS * VIDEO_FPS)  # 90
HOP = WINDOW // 2                          # 45


def _pose_seq(df) -> np.ndarray:
    x = df[POSE_X].to_numpy(dtype=np.float64)
    y = df[POSE_Y].to_numpy(dtype=np.float64)
    return np.stack([x, y], axis=-1)


def make_windows(df, length=WINDOW, hop=HOP) -> dict:
    """Windows from one session or several (grouped by `session`).

    X_pose is normalized per window. IMU is not standardized here.
    Windows with any NaN in the pose are dropped. A window with no finite IMU
    sample keeps imu_mask 0 and zeros.
    """
    if "session" in df.columns and df["session"].nunique() > 1:
        parts = [make_windows(g, length=length, hop=hop) for _, g in df.groupby("session", sort=False)]
        parts = [p for p in parts if len(p["y"])]
        if not parts:
            return _empty()
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    df = df.sort_values("frame").reset_index(drop=True)
    n = len(df)
    pose = _pose_seq(df)
    imu = df[IMU_COLS].to_numpy(dtype=np.float64) if set(IMU_COLS) <= set(df.columns) else np.full((n, 6), np.nan)
    exercise = df["exercise"].to_numpy() if "exercise" in df.columns else np.full(n, "non_activity", dtype=object)
    frames = df["frame"].to_numpy()
    session = str(df["session"].iloc[0]) if "session" in df.columns and n else ""
    split = str(df["split"].iloc[0]) if "split" in df.columns and n else ""

    xs, imus, masks, ys, sessions, starts, splits = [], [], [], [], [], [], []
    if n < length:
        return _empty()
    class_index = {name: i for i, name in enumerate(MMFIT_CLASSES)}
    for start in range(0, n - length + 1, hop):
        sl = slice(start, start + length)
        window = pose[sl]
        if not np.isfinite(window).all():
            continue
        label = exercise[start + length // 2]
        if label not in class_index:
            continue
        norm = normalize(window).reshape(length, 36)
        imu_w = imu[sl]
        finite = np.isfinite(imu_w).all(axis=1)
        if finite.any():
            filled = np.nan_to_num(imu_w, nan=0.0)
            mask = 1.0
        else:
            filled = np.zeros_like(imu_w)
            mask = 0.0
        xs.append(norm.astype(np.float32))
        imus.append(filled.astype(np.float32))
        masks.append(mask)
        ys.append(class_index[label])
        sessions.append(session)
        starts.append(int(frames[start]))
        splits.append(split)
    if not xs:
        return _empty()
    return {
        "X_pose": np.stack(xs),
        "X_imu": np.stack(imus),
        "imu_mask": np.asarray(masks, dtype=np.float32),
        "y": np.asarray(ys, dtype=np.int64),
        "session": np.asarray(sessions),
        "start_frame": np.asarray(starts, dtype=np.int64),
        "split": np.asarray(splits),
    }


def _empty():
    return {
        "X_pose": np.zeros((0, WINDOW, 36), np.float32),
        "X_imu": np.zeros((0, WINDOW, 6), np.float32),
        "imu_mask": np.zeros((0,), np.float32),
        "y": np.zeros((0,), np.int64),
        "session": np.asarray([], dtype=str),
        "start_frame": np.zeros((0,), np.int64),
        "split": np.asarray([], dtype=str),
    }


def _concat(parts):
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def standardize_imu(pack) -> dict:
    """Z-score IMU using mean/std of train windows that have IMU. Mask-0 windows stay 0.

    Writes data/processed/imu_stats.json.
    """
    train = pack["split"] == "train"
    use = train & (pack["imu_mask"] == 1)
    if not use.any():
        use = pack["imu_mask"] == 1
    values = pack["X_imu"][use]
    mean = values.mean(axis=(0, 1)) if len(values) else np.zeros(6, np.float64)
    std = values.std(axis=(0, 1)) if len(values) else np.ones(6, np.float64)
    std = np.maximum(std, 1e-6)
    x = pack["X_imu"].astype(np.float32, copy=True)
    has = pack["imu_mask"] == 1
    x[has] = ((x[has] - mean.astype(np.float32)) / std.astype(np.float32)).astype(np.float32)
    x[~has] = 0
    pack = dict(pack)
    pack["X_imu"] = x
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "imu_stats.json").write_text(json.dumps({
        "mean": mean.tolist(),
        "std": std.tolist(),
    }, indent=2), encoding="utf-8")
    return pack


def build_m1_windows(source=None) -> dict:
    """Read synced Parquet (or a dataframe), cut windows, standardize IMU, save the npz."""
    if source is None:
        parts = []
        root = INTERIM / "mmfit_synced"
        files = sorted(root.glob("session=*"))
        if not files:
            df = pd.read_parquet(root)
            parts.append(make_windows(df))
        else:
            for part in files:
                df = pd.read_parquet(part)
                session = part.name.split("=", 1)[1]
                if "session" not in df.columns:
                    df["session"] = session
                parts.append(make_windows(df))
        pack = _concat([p for p in parts if len(p["y"])])
    elif isinstance(source, pd.DataFrame):
        pack = make_windows(source)
    else:
        pack = make_windows(pd.read_parquet(source))
    pack = standardize_imu(pack)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED / "m1_windows.npz", **pack)
    _print_counts(pack)
    return pack


def _print_counts(pack):
    print(f"windows total {len(pack['y'])}")
    for split in ["train", "val", "test_seen", "test_unseen"]:
        m = pack["split"] == split
        print(f"  {split}: {int(m.sum())}")
        for i, name in enumerate(MMFIT_CLASSES):
            print(f"    {name}: {int((pack['y'][m] == i).sum())}")


def main():
    build_m1_windows()


if __name__ == "__main__":
    main()
