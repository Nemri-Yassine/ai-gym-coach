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


def _pose_from_session(session_dir: Path, set_n: int, person: str, session_folder: str):
    """Prefer cached interim parquet; else run MediaPipe on the set video."""
    cached = OWN_INTERIM / person / f"{session_folder}_set{set_n:02d}.parquet"
    if cached.is_file():
        df = pd.read_parquet(cached)
        xs = df[[f"x{i}" for i in range(18)]].to_numpy(dtype=np.float64)
        ys = df[[f"y{i}" for i in range(18)]].to_numpy(dtype=np.float64)
        pose = np.stack([xs, ys], axis=-1)
        hr = df["hr"].to_numpy(dtype=np.float64) if "hr" in df.columns else None
        hr_mask = df["hr_mask"].to_numpy() if "hr_mask" in df.columns else None
        return pose, hr, hr_mask

    from awc.features.pose_video import extract_pose

    video = session_dir / "sets" / f"set{set_n:02d}.mp4"
    frames_csv = session_dir / f"frames_set{set_n:02d}.csv"
    if not video.is_file() or video.stat().st_size <= 0:
        raise FileNotFoundError(video)
    times = None
    if frames_csv.is_file():
        times = pd.read_csv(frames_csv)["pc_ts_ms"].to_numpy(dtype=np.float64)
    pose, _, _ = extract_pose(video, frame_times_ms=times)
    # HR join
    hr_path = session_dir / "hr.csv"
    hr = hr_mask = None
    if hr_path.is_file() and times is not None and len(times):
        t0 = float(times[0])
        pc = t0 + np.arange(len(pose)) * (1000.0 / VIDEO_FPS)
        rows = [(int(r.pc_ts_ms), float(r.bpm)) for r in pd.read_csv(hr_path).itertuples()]
        hr, hr_mask = join_hr_nearest(pc, rows)
    return pose, hr, hr_mask


def analyze_set(session_dir: Path, set_n: int, person: str | None = None,
                session_folder: str | None = None) -> dict:
    """Run the detection chain for one set. Returns the analysis dict (also written to disk)."""
    t0 = time.perf_counter()
    session_dir = Path(session_dir)
    meta_path = session_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    person = person or meta.get("user") or session_dir.parent.name
    session_folder = session_folder or session_dir.name

    pose, hr, hr_mask = _pose_from_session(session_dir, set_n, person, session_folder)
    good = np.isfinite(pose).all(axis=(1, 2))
    finite = pose[good]
    exercise = "non_activity"
    confidence = 0.0
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
                # confidence = fraction of windows agreeing with the vote (excl. non_activity pool)
                confidence = float((preds == vote).mean())

    n_reps = 0
    form_score = None
    if len(finite) >= 16 and exercise != "non_activity":
        pose_n = normalize(finite)
        n_reps, _ = count_reps(pose_n, exercise=exercise)
        form_score = _m2_mean(exercise, pose_n)

    hr_start = hr_end = hr_mean = None
    if hr is not None and hr_mask is not None:
        vals = hr[hr_mask.astype(bool) & np.isfinite(hr)]
        if len(vals):
            hr_mean = float(np.mean(vals))
            hr_start = float(vals[0])
            hr_end = float(vals[-1])

    elapsed = time.perf_counter() - t0
    result = {
        "person": person,
        "session_folder": session_folder,
        "set": int(set_n),
        "exercise": exercise,
        "confidence": round(confidence, 3),
        "reps": int(n_reps),
        "form_score": None if form_score is None else round(float(form_score), 3),
        "hr_start": None if hr_start is None else round(hr_start, 1),
        "hr_end": None if hr_end is None else round(hr_end, 1),
        "hr_mean": None if hr_mean is None else round(hr_mean, 1),
        "elapsed_s": round(elapsed, 2),
        "summary": _format_summary(exercise, confidence, n_reps, form_score, hr_start, hr_end, elapsed),
    }
    out = analysis_path(person, session_folder, set_n)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _format_summary(exercise, confidence, reps, form_score, hr_start, hr_end, elapsed):
    form = "—" if form_score is None else f"{form_score:.2f}"
    if hr_start is not None and hr_end is not None:
        hr = f"HR {hr_start:.0f} → {hr_end:.0f} bpm"
    else:
        hr = "HR —"
    return f"Model: {exercise} ({confidence:.2f}) · {reps} reps counted · form: {form} · {hr} · {elapsed:.1f} s"


def schedule_analysis(session_dir: Path, set_n: int, on_done=None, kafka_publish=None):
    """Fire-and-forget background worker. on_done(result) called when finished."""
    import threading

    def worker():
        try:
            result = analyze_set(session_dir, set_n)
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

    threading.Thread(target=worker, daemon=True, name=f"awc-analyze-{set_n}").start()
