"""MM-Fit RGB video → MediaPipe pose on the sensor frame clock + watch join.

    python scripts/process_mmfit_video.py w20
    python scripts/process_mmfit_video.py w20 --fit-sync

Writes data/interim/mmfit_video/<session>.parquet (labelled set frames only by default).
Never writes under data/raw/own_recordings/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from awc.config import (
    MMFIT_VIDEO_DIR,
    MMFIT_VIDEO_INTERIM,
    VIDEO_FPS,
    VIDEO_SYNC_PATH,
)
from awc.data import mmfit
from awc.features.pose_video import extract_pose_ranges
from awc.features.skeleton import normalize
from awc.features.sync import (
    IMU_COLS,
    POSE_X,
    POSE_Y,
    attach_labels,
    hr_to_frame,
    imu_to_frame,
    session_split,
)
from awc.features.video_sync import (
    cross_corr_offset,
    default_w20_fit,
    fit_session_from_set_offsets,
    get_session_fit,
    load_sync,
    midhip_height,
    save_sync,
    sensor_from_video,
    video_from_sensor,
)


def _dataset_pose_y_up(session_id):
    """(T, 18, 2) MM-Fit pose_2d with y up, indexed by sensor frame number in column 0."""
    raw = np.array(mmfit.load_stream(session_id, "pose_2d", mmap=False))
    frames = np.rint(raw[0, :, 0]).astype(np.int64)
    xs = raw[0, :, 1:19].astype(np.float64)
    ys = -raw[1, :, 1:19].astype(np.float64)
    pose = np.stack([xs, ys], axis=-1)
    return frames, pose


def fit_sync_for_session(session_id, video_path, max_sets=12):
    """Cross-correlate MediaPipe mid-hip with dataset pose on labelled sets; fit a, b."""
    labels = mmfit.load_labels(session_id)
    sensor_frames, dataset_pose = _dataset_pose_y_up(session_id)
    frame_to_idx = {int(f): i for i, f in enumerate(sensor_frames)}

    # Approximate video ranges with identity (or prior fit) then refine.
    prior = default_w20_fit() if session_id == "w20" else {"a": 0.0, "b": 0.0, "pivot": 0.0}
    approx_ranges = []
    meta = []
    for i, (s0, s1, reps, ex) in enumerate(labels[:max_sets]):
        v0 = int(round(video_from_sensor(s0, prior["a"], prior["b"], prior["pivot"])))
        v1 = int(round(video_from_sensor(s1, prior["a"], prior["b"], prior["pivot"])))
        v0 = max(0, v0 - 30)
        v1 = v1 + 30
        approx_ranges.append((v0, v1))
        meta.append((i, s0, s1, ex, v0, v1))

    mp_by_range = extract_pose_ranges(video_path, approx_ranges)
    video_anchors, sensor_anchors = [], []
    for (i, s0, s1, ex, v0, v1), key in zip(meta, approx_ranges):
        mp = mp_by_range.get(key)
        if mp is None or len(mp) < 30:
            continue
        idxs = [frame_to_idx[f] for f in range(s0, s1 + 1) if f in frame_to_idx]
        if len(idxs) < 30:
            continue
        ref = midhip_height(dataset_pose[idxs])
        # MediaPipe clip starts at v0; map to sensor via lag at set start.
        query = midhip_height(mp)
        # Trim to comparable lengths
        n = min(len(ref), len(query))
        lag = cross_corr_offset(ref[:n], query[:n], max_lag=90)
        # query[t] ≈ ref[t + lag] → video frame (v0 + t) maps near sensor (s0 + t + lag)
        # Use set-start anchors: video start of movement ≈ v0, sensor ≈ s0 + lag
        video_anchors.append(float(v0))
        sensor_anchors.append(float(s0 + lag))

    if len(video_anchors) < 3:
        raise RuntimeError(f"{session_id}: only {len(video_anchors)} sync anchors")
    fit = fit_session_from_set_offsets(video_anchors, sensor_anchors)
    fit["session"] = session_id
    fit["source"] = "cross_corr_midhip"
    return fit


def _pose_rows(sensor_frame_ids, pose_t18):
    rows = {"frame": np.asarray(sensor_frame_ids, dtype=np.int64)}
    for j in range(18):
        rows[f"x{j}"] = pose_t18[:, j, 0]
        rows[f"y{j}"] = pose_t18[:, j, 1]
    return pd.DataFrame(rows)


def process_session(session_id, fit_sync=False, max_sync_sets=12):
    video_path = MMFIT_VIDEO_DIR / f"{session_id}_rgb.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    all_fits = load_sync()
    if fit_sync or session_id not in all_fits:
        if fit_sync or session_id != "w20":
            print(f"fitting sync for {session_id}…", flush=True)
            fit = fit_sync_for_session(session_id, video_path, max_sets=max_sync_sets)
        else:
            fit = default_w20_fit()
            fit["session"] = session_id
            fit["n_sets"] = None
            fit["n_within_3"] = None
            fit["frac_within_3"] = None
        all_fits[session_id] = fit
        save_sync(all_fits)
        print(json.dumps(fit, indent=2), flush=True)
    else:
        fit = all_fits[session_id]

    labels = mmfit.load_labels(session_id)
    # Video ranges covering each labelled set (with small pad), then map to sensor.
    video_ranges = []
    for s0, s1, reps, ex in labels:
        v0 = int(np.floor(video_from_sensor(s0, fit["a"], fit["b"], fit["pivot"]))) - 5
        v1 = int(np.ceil(video_from_sensor(s1, fit["a"], fit["b"], fit["pivot"]))) + 5
        video_ranges.append((max(0, v0), max(v0 + 1, v1)))

    print(f"MediaPipe on {len(video_ranges)} sets…", flush=True)
    mp_clips = extract_pose_ranges(video_path, video_ranges)

    # Build a dense sensor-frame table over labelled frames only.
    sensor_ids = []
    pose_list = []
    for (s0, s1, reps, ex), (v0, v1) in zip(labels, video_ranges):
        clip = mp_clips.get((v0, v1))
        if clip is None:
            continue
        for local, video_f in enumerate(range(v0, v1 + 1)):
            if local >= len(clip):
                break
            sens = int(round(sensor_from_video(video_f, fit["a"], fit["b"], fit["pivot"])))
            if sens < s0 or sens > s1:
                continue
            sensor_ids.append(sens)
            pose_list.append(clip[local])

    if not pose_list:
        raise RuntimeError(f"{session_id}: no MediaPipe frames mapped into labelled sets")

    pose_arr = np.asarray(pose_list, dtype=np.float64)
    # If duplicate sensor frames, keep last
    df = _pose_rows(sensor_ids, pose_arr)
    df = df.drop_duplicates("frame", keep="last").sort_values("frame").reset_index(drop=True)

    # Join watch IMU + HR on sensor frames
    for stream, prefix in (("sw_l_acc", "acc"), ("sw_l_gyr", "gyr")):
        raw = mmfit.load_stream(session_id, stream, mmap=False)
        if raw is None:
            for axis in "xyz":
                df[f"{prefix}_{axis}"] = np.nan
            continue
        df = df.merge(imu_to_frame(np.array(raw), prefix), on="frame", how="left")
    hr_raw = mmfit.load_stream(session_id, "sw_l_hr", mmap=False)
    if hr_raw is None:
        hr_raw = mmfit.load_stream(session_id, "sw_r_hr", mmap=False)
    hr = hr_to_frame(None if hr_raw is None else np.array(hr_raw), df["frame"].to_numpy())
    df["hr"] = hr["hr"].to_numpy()
    df["hr_mask"] = hr["hr_mask"].to_numpy()
    df = attach_labels(df, labels)
    df["session"] = session_id
    df["split"] = session_split(session_id)
    df["pose_source"] = "mediapipe"

    # Normalized pose stored as flat columns nx0.. for convenience of RQ4
    # (normalize per set later in eval; keep pixels here).
    MMFIT_VIDEO_INTERIM.mkdir(parents=True, exist_ok=True)
    out = MMFIT_VIDEO_INTERIM / f"{session_id}.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} rows={len(df)}", flush=True)
    return out, fit


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("session", help="e.g. w20")
    p.add_argument("--fit-sync", action="store_true", help="Re-fit video↔sensor map")
    p.add_argument("--max-sync-sets", type=int, default=12)
    args = p.parse_args(argv)
    process_session(args.session, fit_sync=args.fit_sync, max_sync_sets=args.max_sync_sets)


if __name__ == "__main__":
    main()
