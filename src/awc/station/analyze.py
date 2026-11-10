"""Post-set analysis for the gym station (pose → M1 → reps → M2 → HR).

Runs in a background thread after video_saved. Writes JSON under
data/interim/own_live/<person>/<session>/analysis_setNN.json — never under data/raw.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from awc.config import MMFIT_CLASSES, MODELS, OWN_DIR, OWN_INTERIM, OWN_LIVE_DIR, VIDEO_FPS
from awc.features.own_sync import join_hr_nearest
from awc.features.reps import count_reps, rep_boundaries
from awc.features.skeleton import normalize
from awc.models.m1_recognition import M1Model

log = logging.getLogger("awc.station.analyze")

WINDOW, HOP = 90, 45
PRIMARY = "m1_transformer_pose"
# Own-camera M2 is uncalibrated (P(correct)≈0.06 on correct squats). Never show "bad form".
FORM_CALIBRATED = False
# Live suggestion only: every 2nd frame (~15 fps) + 480px long side to keep MediaPipe under ~15 s.
LIVE_FRAME_STRIDE = 2
LIVE_MAX_SIDE = 480


def analysis_path(person: str, session_folder: str, set_n: int) -> Path:
    return OWN_LIVE_DIR / person / session_folder / f"analysis_set{set_n:02d}.json"


def _load_m1():
    path = MODELS / f"{PRIMARY}.pt"
    if not path.is_file():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = M1Model(ckpt["arch"], ckpt["inputs"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def _vote(preds):
    from collections import Counter
    na = MMFIT_CLASSES.index("non_activity")
    active = [int(p) for p in preds if p != na]
    pool = active or [int(p) for p in preds]
    return Counter(pool).most_common(1)[0][0] if pool else na


def _windows(pose):
    out = []
    for s in range(0, len(pose) - WINDOW + 1, HOP):
        w = pose[s:s + WINDOW]
        if np.isfinite(w).all():
            out.append(normalize(w).reshape(WINDOW, 36))
    return np.asarray(out, dtype=np.float32).reshape(-1, WINDOW, 36)


def _m2_mean(exercise: str, pose_norm: np.ndarray) -> float | None:
    if exercise not in ("squats", "lunges", "plank"):
        return None
    path = MODELS / "m2_rf.pickle"
    if not path.is_file():
        return None
    import pickle
    from awc.features.form_dataset import resample
    from awc.features.form_features import feature_vector, features_2d

    with open(path, "rb") as f:
        pack = pickle.load(f)
    ex_key = {"squats": "squat", "lunges": "lunge", "plank": "plank"}[exercise]
    forest = pack.get("forests", {}).get(ex_key)
    if forest is None:
        return None
    bounds = rep_boundaries(pose_norm, exercise=exercise)
    probs = []
    for a, b in bounds:
        seg = pose_norm[a:b + 1]
        if len(seg) < 8 or not np.isfinite(seg).all():
            continue
        try:
            feats = feature_vector(features_2d(resample(seg, 64))).reshape(1, -1)
            proba = forest.predict_proba(feats)[0]
            classes = list(forest.classes_)
            probs.append(float(proba[classes.index(0)]) if 0 in classes else float(proba.max()))
        except Exception:  # noqa: BLE001
            continue
    return float(np.mean(probs)) if probs else None


def _extract_pose_live(video: Path, frames_csv: Path | None, frame_times_ms,
                       frame_stride: int = LIVE_FRAME_STRIDE, max_side: int = LIVE_MAX_SIDE):
    """MediaPipe for the live suggestion: subsample frames + optional downscale."""
    import cv2
    from awc.features.pose_video import (
        MAX_GAP_SECONDS,
        _detect_bgr,
        _open_landmarker,
        fill_gaps,
        resample,
        resample_times,
    )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    if not np.isfinite(fps) or fps <= 1:
        fps = VIDEO_FPS
    frames = []
    kept_times = []
    with _open_landmarker() as landmarker:
        index = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if index % max(1, int(frame_stride)) != 0:
                index += 1
                continue
            h, w = bgr.shape[:2]
            long_side = max(h, w)
            if max_side and long_side > max_side:
                scale = max_side / float(long_side)
                bgr = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))),
                                 interpolation=cv2.INTER_AREA)
            if frame_times_ms is not None and index < len(frame_times_ms):
                ts = int(frame_times_ms[index] - frame_times_ms[0])
                kept_times.append(float(frame_times_ms[index]))
            else:
                ts = int(round(index * 1000 / fps))
                kept_times.append(float(ts))
            pose, _ = _detect_bgr(landmarker, bgr, ts, return_visibility=True)
            frames.append(pose)
            index += 1
    cap.release()
    raw = np.asarray(frames, dtype=np.float64).reshape(-1, 18, 2)
    if frame_times_ms is not None and kept_times:
        times = np.asarray(kept_times, dtype=np.float64)
        real_fps = (len(times) - 1) / max((times[-1] - times[0]) / 1000.0, 1e-6)
        filled = fill_gaps(raw, max_gap=int(round(MAX_GAP_SECONDS * real_fps)))
        pose = resample_times(filled, times).astype(np.float32)
    else:
        src_fps = float(fps) / max(1, int(frame_stride))
        filled = fill_gaps(raw, max_gap=int(round(MAX_GAP_SECONDS * src_fps)))
        pose = resample(filled, src_fps).astype(np.float32)
    return pose


def _pose_from_session(session_dir: Path, set_n: int, person: str, session_folder: str,
                       use_cache: bool = True, live_fast: bool = False):
    """Load pose. use_cache=False forces MediaPipe on the set video (honest live latency)."""
    cached = OWN_INTERIM / person / f"{session_folder}_set{set_n:02d}.parquet"
    if use_cache and cached.is_file():
        df = pd.read_parquet(cached)
        xs = df[[f"x{i}" for i in range(18)]].to_numpy(dtype=np.float64)
        ys = df[[f"y{i}" for i in range(18)]].to_numpy(dtype=np.float64)
        pose = np.stack([xs, ys], axis=-1)
        hr = df["hr"].to_numpy(dtype=np.float64) if "hr" in df.columns else None
        hr_mask = df["hr_mask"].to_numpy() if "hr_mask" in df.columns else None
        return pose, hr, hr_mask, "cache"

    video = session_dir / "sets" / f"set{set_n:02d}.mp4"
    frames_csv = session_dir / f"frames_set{set_n:02d}.csv"
    if not video.is_file() or video.stat().st_size <= 0:
        raise FileNotFoundError(video)
    times = None
    if frames_csv.is_file():
        times = pd.read_csv(frames_csv)["pc_ts_ms"].to_numpy(dtype=np.float64)

    if live_fast:
        pose = _extract_pose_live(video, frames_csv, times)
        source = f"mediapipe_stride{LIVE_FRAME_STRIDE}_max{LIVE_MAX_SIDE}"
    else:
        from awc.features.pose_video import extract_pose
        pose, _, _ = extract_pose(video, frame_times_ms=times)
        source = "mediapipe_full"

    hr = hr_mask = None
    if frames_csv.is_file() and times is not None and len(times):
        hr_path = session_dir / "hr.csv"
        if hr_path.is_file():
            t0 = float(times[0])
            pc = t0 + np.arange(len(pose)) * (1000.0 / VIDEO_FPS)
            rows = [(int(r.pc_ts_ms), float(r.bpm)) for r in pd.read_csv(hr_path).itertuples()]
            hr, hr_mask = join_hr_nearest(pc, rows)
    return pose, hr, hr_mask, source


def analyze_set(session_dir: Path, set_n: int, person: str | None = None,
                session_folder: str | None = None, *,
                use_cache: bool = True, live_fast: bool | None = None,
                show_form: bool = False) -> dict:
    """Run the detection chain for one set. Returns the analysis dict (also written to disk).

    use_cache=False bypasses interim parquet (honest live latency, includes MediaPipe).
    live_fast defaults to True when use_cache is False (15 fps + 480px for the suggestion only).
    show_form=False hides M2 form scores until the model is calibrated for this camera.
    """
    t_all = time.perf_counter()
    session_dir = Path(session_dir)
    meta_path = session_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    person = person or meta.get("user") or session_dir.parent.name
    session_folder = session_folder or session_dir.name
    if live_fast is None:
        live_fast = not use_cache

    t0 = time.perf_counter()
    pose, hr, hr_mask, pose_source = _pose_from_session(
        session_dir, set_n, person, session_folder, use_cache=use_cache, live_fast=live_fast,
    )
    t_pose = time.perf_counter() - t0

    good = np.isfinite(pose).all(axis=(1, 2))
    finite = pose[good]
    exercise = "non_activity"
    confidence = 0.0
    t0 = time.perf_counter()
    model = _load_m1()
    if model is not None and len(finite) >= WINDOW:
        x = _windows(finite)
        if len(x):
            with torch.no_grad():
                if model.inputs == "pose":
                    logits = model(torch.from_numpy(x))
                else:
                    logits = model(torch.from_numpy(x),
                                   torch.zeros(len(x), WINDOW, 6),
                                   torch.zeros(len(x)))
                preds = logits.argmax(1).numpy()
                vote = _vote(preds)
                exercise = MMFIT_CLASSES[vote]
                confidence = float((preds == vote).mean())
    t_m1 = time.perf_counter() - t0

    n_reps = 0
    form_score = None
    t0 = time.perf_counter()
    pose_n = None
    if len(finite) >= 16 and exercise != "non_activity":
        pose_n = normalize(finite)
        n_reps, _ = count_reps(pose_n, exercise=exercise)
    t_reps = time.perf_counter() - t0

    t0 = time.perf_counter()
    if pose_n is not None and (show_form or FORM_CALIBRATED):
        form_score = _m2_mean(exercise, pose_n)
    t_m2 = time.perf_counter() - t0

    hr_start = hr_end = hr_mean = None
    if hr is not None and hr_mask is not None:
        vals = hr[hr_mask.astype(bool) & np.isfinite(hr)]
        if len(vals):
            hr_mean = float(np.mean(vals))
            hr_start = float(vals[0])
            hr_end = float(vals[-1])

    elapsed = time.perf_counter() - t_all
    timings = {
        "pose_s": round(t_pose, 3),
        "m1_s": round(t_m1, 3),
        "reps_s": round(t_reps, 3),
        "m2_s": round(t_m2, 3),
        "total_s": round(elapsed, 3),
        "pose_source": pose_source,
        "use_cache": bool(use_cache),
        "live_fast": bool(live_fast),
    }
    result = {
        "person": person,
        "session_folder": session_folder,
        "set": int(set_n),
        "exercise": exercise,
        "confidence": round(confidence, 3),
        "reps": int(n_reps),
        "form_score": None if form_score is None else round(float(form_score), 3),
        "form_shown": bool(show_form or FORM_CALIBRATED),
        "form_note": None if (show_form or FORM_CALIBRATED) else (
            "form model: not calibrated for this camera (P(correct) 0.06 on our correct squats)"
        ),
        "hr_start": None if hr_start is None else round(hr_start, 1),
        "hr_end": None if hr_end is None else round(hr_end, 1),
        "hr_mean": None if hr_mean is None else round(hr_mean, 1),
        "elapsed_s": round(elapsed, 2),
        "timings": timings,
        "summary": _format_summary(
            exercise, confidence, n_reps, form_score, hr_start, hr_end, elapsed,
            show_form=show_form or FORM_CALIBRATED,
        ),
    }
    out = analysis_path(person, session_folder, set_n)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _format_summary(exercise, confidence, reps, form_score, hr_start, hr_end, elapsed,
                    show_form: bool = False):
    if show_form:
        form = "—" if form_score is None else f"{form_score:.2f}"
        form_part = f"form: {form}"
    else:
        form_part = "form model: not calibrated for this camera (P(correct) 0.06 on our correct squats)"
    if hr_start is not None and hr_end is not None:
        hr = f"HR {hr_start:.0f} → {hr_end:.0f} bpm"
    else:
        hr = "HR —"
    return (
        f"Model: {exercise} ({confidence:.2f}) · {reps} reps counted · "
        f"{form_part} · {hr} · {elapsed:.1f} s"
    )


def schedule_analysis(session_dir: Path, set_n: int, on_done=None, kafka_publish=None,
                      show_form: bool = False):
    """Fire-and-forget background worker. on_done(result) called when finished.

    Live path uses use_cache=False + live_fast MediaPipe so latency includes pose.
    """
    import threading

    def worker():
        try:
            # Live path always runs MediaPipe (live_fast). Cached parquet is for offline
            # eval / measure_live_latency.py comparisons — never for the station suggestion.
            result = analyze_set(
                session_dir, set_n,
                use_cache=False,
                live_fast=True,
                show_form=show_form,
            )
            if kafka_publish is not None:
                try:
                    kafka_publish(result)
                except Exception as exc:  # noqa: BLE001
                    log.warning("kafka publish failed: %s", exc)
            if on_done is not None:
                on_done(result)
        except Exception as exc:  # noqa: BLE001
            log.exception("analysis failed for set %s: %s", set_n, exc)
            if on_done is not None:
                on_done({"set": set_n, "error": str(exc), "summary": f"analysis failed: {exc}"})

    threading.Thread(target=worker, daemon=True).start()
