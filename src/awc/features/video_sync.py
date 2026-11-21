"""Fit video_frame → sensor_frame linear maps per MM-Fit session.

sensor_frame = video_frame + a + b * (video_frame - pivot)

Cross-correlate a MediaPipe signal (mid-hip height) with the MM-Fit pose_2d
signal on labelled sets, then least-squares fit (a, b). Used for RQ4 only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from awc.config import INTERIM, VIDEO_FPS, VIDEO_SYNC_PATH, W20_VIDEO_SENSOR_OFFSET, W20_VIDEO_SENSOR_PIVOT, W20_VIDEO_SENSOR_SLOPE


def midhip_height(pose_t18x2: np.ndarray) -> np.ndarray:
    """Mean of left/right hip y. pose is (T, 18, 2), y up."""
    p = np.asarray(pose_t18x2, dtype=np.float64)
    return 0.5 * (p[:, 8, 1] + p[:, 11, 1])


def cross_corr_offset(ref: np.ndarray, query: np.ndarray, max_lag: int = 90) -> int:
    """Integer lag that best aligns query onto ref: query[t] ≈ ref[t + lag].

    Positive lag means query is early relative to ref (query events happen at
    smaller indices than the matching ref events).
    """
    a = np.asarray(ref, dtype=np.float64)
    b = np.asarray(query, dtype=np.float64)
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    a = np.nan_to_num(a, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)
    if len(a) < 8 or len(b) < 8:
        return 0
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-len(b) + 1, len(a))
    keep = (lags >= -max_lag) & (lags <= max_lag)
    corr = corr[keep]
    lags = lags[keep]
    return int(lags[int(np.argmax(corr))])


def fit_linear_sync(video_frames: np.ndarray, sensor_frames: np.ndarray, pivot: float | None = None):
    """Least squares: sensor = video + a + b * (video - pivot). Returns a, b, pivot."""
    v = np.asarray(video_frames, dtype=np.float64)
    s = np.asarray(sensor_frames, dtype=np.float64)
    if pivot is None:
        pivot = float(np.median(v))
    x = v - pivot
    # s - v = a + b * x
    y = s - v
    # design [1, x]
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return a, b, float(pivot)


def sensor_from_video(video_frame, a, b, pivot) -> float:
    v = float(video_frame)
    return v + a + b * (v - pivot)


def video_from_sensor(sensor_frame, a, b, pivot) -> float:
    s = float(sensor_frame)
    return (s - a + b * pivot) / (1.0 + b)


def within_frames(pred, truth, tol=3) -> bool:
    return abs(float(pred) - float(truth)) <= tol


def fit_session_from_set_offsets(set_video_starts, set_sensor_starts, pivot=None):
    """Fit from paired set-start estimates (video index, sensor index)."""
    v = np.asarray(set_video_starts, dtype=np.float64)
    s = np.asarray(set_sensor_starts, dtype=np.float64)
    a, b, pivot = fit_linear_sync(v, s, pivot=pivot)
    preds = [sensor_from_video(vi, a, b, pivot) for vi in v]
    n_ok = sum(within_frames(p, t, 3) for p, t in zip(preds, s))
    return {
        "a": a,
        "b": b,
        "pivot": pivot,
        "n_sets": int(len(v)),
        "n_within_3": int(n_ok),
        "frac_within_3": float(n_ok / max(len(v), 1)),
        "residuals": [float(p - t) for p, t in zip(preds, s)],
    }


def default_w20_fit():
    return {
        "a": float(W20_VIDEO_SENSOR_OFFSET),
        "b": float(W20_VIDEO_SENSOR_SLOPE),
        "pivot": float(W20_VIDEO_SENSOR_PIVOT),
        "source": "config_w20_measured",
    }


def load_sync(path: Path = VIDEO_SYNC_PATH) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_sync(data: dict, path: Path = VIDEO_SYNC_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_session_fit(session_id: str, path: Path = VIDEO_SYNC_PATH) -> dict:
    all_fits = load_sync(path)
    if session_id in all_fits:
        return all_fits[session_id]
    if session_id == "w20":
        return default_w20_fit()
    raise KeyError(f"no video sync fit for {session_id}; run fit first")
