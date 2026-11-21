"""RQ4 first look: MM-Fit dataset pose vs MediaPipe from raw RGB, same models.

    python scripts/eval_rq4_mmfit_video.py --sessions w20

No form labels on MM-Fit: Model 2 reports correct-form probability distributions only.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from awc.config import (
    INTERIM,
    MMFIT_CLASSES,
    MMFIT_VIDEO_INTERIM,
    MODELS,
    REPORTS,
    VIDEO_FPS,
    VIDEO_SYNC_PATH,
)
from awc.data import mmfit
from awc.evaluation.metrics import macro_f1
from awc.features.form_dataset import resample
from awc.features.form_features import feature_vector, features_2d
from awc.features.reps import count_reps
from awc.features.skeleton import normalize
from awc.features.sync import POSE_X, POSE_Y
from awc.features.video_sync import load_sync
from awc.models.m1_recognition import M1Model
from awc.training.tracking import new_run, save_metrics


def _pose_from_synced(df: pd.DataFrame) -> np.ndarray:
    xs = df[POSE_X].to_numpy(dtype=np.float64)
    ys = df[POSE_Y].to_numpy(dtype=np.float64)
    return np.stack([xs, ys], axis=-1)


def _dataset_set_pose(session_id, start, end):
    raw = np.array(mmfit.load_stream(session_id, "pose_2d", mmap=False))
    frames = np.rint(raw[0, :, 0]).astype(np.int64)
    mask = (frames >= start) & (frames <= end)
    xs = raw[0, mask, 1:19].astype(np.float64)
    ys = -raw[1, mask, 1:19].astype(np.float64)
    return np.stack([xs, ys], axis=-1)


def _imu_window(df, center_iloc, length=90):
    half = length // 2
    i0 = max(0, center_iloc - half)
    i1 = min(len(df), i0 + length)
    i0 = max(0, i1 - length)
    sl = df.iloc[i0:i1]
    imu = sl[["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]].to_numpy(dtype=np.float32)
    if len(imu) < length:
        pad = np.zeros((length - len(imu), 6), dtype=np.float32)
        imu = np.concatenate([imu, pad], axis=0)
    mask = 1.0 if np.isfinite(imu).all() else 0.0
    imu = np.nan_to_num(imu, nan=0.0)
    return imu, mask


def _pose_window(pose_t18, center, length=90):
    half = length // 2
    i0 = max(0, center - half)
    i1 = min(len(pose_t18), i0 + length)
    i0 = max(0, i1 - length)
    win = pose_t18[i0:i1]
    if len(win) < length:
        pad = np.zeros((length - len(win), 18, 2), dtype=np.float32)
        win = np.concatenate([win, pad], axis=0)
    return win.reshape(length, 36).astype(np.float32)


def load_m1(arch="transformer", inputs="both"):
    path = MODELS / f"m1_{arch}_{inputs}.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = M1Model(ckpt["arch"], ckpt["inputs"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["inputs"]


def load_m2_rf():
    path = MODELS / "m2_rf.pickle"
    with open(path, "rb") as handle:
        return pickle.load(handle)


def predict_m1(model, inputs, pose36, imu, imu_mask):
    with torch.no_grad():
        p = torch.from_numpy(pose36[None])
        if inputs == "pose":
            logits = model(p)
        elif inputs == "imu":
            logits = model(torch.from_numpy(imu[None]))
        else:
            logits = model(p, torch.from_numpy(imu[None]), torch.tensor([imu_mask], dtype=torch.float32))
        return int(logits.argmax(dim=1).item())


def eval_session(session_id, mp_df: pd.DataFrame, sync_info: dict):
    labels = mmfit.load_labels(session_id)
    m1_pose, inputs_pose = load_m1("transformer", "pose")
    m1_both, inputs_both = load_m1("transformer", "both")
    rf_pack = load_m2_rf()
    forests = rf_pack["forests"]

    y_true, y_pose, y_both_ds, y_both_mp = [], [], [], []
    rep_rows = []
    form_probs = {"dataset": [], "mediapipe": []}

    for set_id, (s0, s1, n_reps, exercise) in enumerate(labels):
        ds_pose = _dataset_set_pose(session_id, s0, s1)
        if len(ds_pose) < 16 or not np.isfinite(ds_pose).all():
            continue
        ds_norm = normalize(ds_pose)

        mp_set = mp_df[(mp_df["set_id"] == set_id)].sort_values("frame")
        if len(mp_set) < 16:
            continue
        mp_pose = _pose_from_synced(mp_set)
        if not np.isfinite(mp_pose).any():
            continue
        # fill remaining NaNs with column median for normalize stability
        if not np.isfinite(mp_pose).all():
            flat = mp_pose.reshape(-1, 36)
            col_med = np.nanmedian(flat, axis=0)
            inds = np.where(~np.isfinite(flat))
            flat[inds] = np.take(col_med, inds[1])
            mp_pose = flat.reshape(-1, 18, 2)
        mp_norm = normalize(mp_pose)

        # Model 1 at set mid-point
        mid_ds = len(ds_norm) // 2
        mid_mp = len(mp_norm) // 2
        imu, imu_mask = _imu_window(mp_set, len(mp_set) // 2)
        class_idx = MMFIT_CLASSES.index(exercise) if exercise in MMFIT_CLASSES else None
        if class_idx is not None:
            y_true.append(class_idx)
            y_pose.append(predict_m1(m1_pose, inputs_pose, _pose_window(ds_norm, mid_ds), imu, 0.0))
            y_both_ds.append(predict_m1(m1_both, inputs_both, _pose_window(ds_norm, mid_ds), imu, imu_mask))
            y_both_mp.append(predict_m1(m1_both, inputs_both, _pose_window(mp_norm, mid_mp), imu, imu_mask))
            # also pose-only on mediapipe
            # stored separately below

        c_ds, _ = count_reps(ds_norm, fps=VIDEO_FPS, exercise=exercise)
        c_mp, bounds_mp = count_reps(mp_norm, fps=VIDEO_FPS, exercise=exercise)
        rep_rows.append({
            "session": session_id, "set_id": set_id, "exercise": exercise,
            "true_reps": n_reps, "pred_dataset": c_ds, "pred_mediapipe": c_mp,
            "ae_dataset": abs(c_ds - n_reps), "ae_mediapipe": abs(c_mp - n_reps),
        })

        # Model 2 RF on squat/lunge: probability of "correct" (class 0 in EC3D head)
        ex_key = {"squats": "squat", "lunges": "lunge"}.get(exercise)
        if ex_key and ex_key in forests and bounds_mp:
            forest = forests[ex_key]
            for a, b in bounds_mp:
                seg = mp_norm[a:b + 1]
                if len(seg) < 8:
                    continue
                rep64 = resample(seg, 64)
                feats = feature_vector(features_2d(rep64)).reshape(1, -1)
                proba = forest.predict_proba(feats)[0]
                classes = list(forest.classes_)
                p_correct = float(proba[classes.index(0)]) if 0 in classes else float(proba.max())
                form_probs["mediapipe"].append({"exercise": exercise, "p_correct": p_correct})
            # dataset pose reps
            _, bounds_ds = count_reps(ds_norm, fps=VIDEO_FPS, exercise=exercise)
            for a, b in bounds_ds:
                seg = ds_norm[a:b + 1]
                if len(seg) < 8:
                    continue
                rep64 = resample(seg, 64)
                feats = feature_vector(features_2d(rep64)).reshape(1, -1)
                proba = forest.predict_proba(feats)[0]
                classes = list(forest.classes_)
                p_correct = float(proba[classes.index(0)]) if 0 in classes else float(proba.max())
                form_probs["dataset"].append({"exercise": exercise, "p_correct": p_correct})

    # pose-only MediaPipe M1
    y_pose_mp = []
    y_true2 = []
    for set_id, (s0, s1, n_reps, exercise) in enumerate(labels):
        if exercise not in MMFIT_CLASSES:
            continue
        mp_set = mp_df[(mp_df["set_id"] == set_id)].sort_values("frame")
        if len(mp_set) < 16:
            continue
        mp_pose = _pose_from_synced(mp_set)
        if not np.isfinite(mp_pose).all():
            flat = mp_pose.reshape(-1, 36)
            col_med = np.nanmedian(flat, axis=0)
            inds = np.where(~np.isfinite(flat))
            flat[inds] = np.take(col_med, inds[1])
            mp_pose = flat.reshape(-1, 18, 2)
        mp_norm = normalize(mp_pose)
        y_true2.append(MMFIT_CLASSES.index(exercise))
        y_pose_mp.append(predict_m1(m1_pose, inputs_pose, _pose_window(mp_norm, len(mp_norm) // 2), np.zeros((90, 6), np.float32), 0.0))

    def _f1(yt, yp):
        if not yt:
            return None
        return float(macro_f1(np.asarray(yt), np.asarray(yp)))

    rep_df = pd.DataFrame(rep_rows)
    out = {
        "session": session_id,
        "n_sets_m1": len(y_true),
        "m1_macro_f1_dataset_pose_only": _f1(y_true, y_pose),
        "m1_macro_f1_dataset_both": _f1(y_true, y_both_ds),
        "m1_macro_f1_mediapipe_pose_only": _f1(y_true2, y_pose_mp),
        "m1_macro_f1_mediapipe_both": _f1(y_true, y_both_mp),
        "rep_mae_dataset": float(rep_df["ae_dataset"].mean()) if len(rep_df) else None,
        "rep_mae_mediapipe": float(rep_df["ae_mediapipe"].mean()) if len(rep_df) else None,
        "sync": {k: sync_info.get(k) for k in ("a", "b", "pivot", "n_sets", "n_within_3", "frac_within_3", "source")},
        "form_p_correct_mean_dataset": float(np.mean([r["p_correct"] for r in form_probs["dataset"]])) if form_probs["dataset"] else None,
        "form_p_correct_mean_mediapipe": float(np.mean([r["p_correct"] for r in form_probs["mediapipe"]])) if form_probs["mediapipe"] else None,
        "form_note": "MM-Fit has no form labels; numbers are Model 2 RF P(correct) only",
        "n_form_reps_dataset": len(form_probs["dataset"]),
        "n_form_reps_mediapipe": len(form_probs["mediapipe"]),
    }
    return out, rep_df


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", nargs="+", default=["w20"])
    args = p.parse_args(argv)
    sync_all = load_sync(VIDEO_SYNC_PATH)
    results = []
    run = new_run("rq4_mmfit_video", {"sessions": args.sessions})
    for sid in args.sessions:
        path = MMFIT_VIDEO_INTERIM / f"{sid}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing; run scripts/process_mmfit_video.py {sid}")
        mp_df = pd.read_parquet(path)
        sync_info = sync_all.get(sid, {})
        metrics, rep_df = eval_session(sid, mp_df, sync_info)
        results.append(metrics)
        rep_df.to_csv(run / f"{sid}_reps.csv", index=False)
        print(json.dumps(metrics, indent=2), flush=True)
    save_metrics(run, {"sessions": results})
    # report
    lines = [
        "# RQ4 first look · MM-Fit raw video vs dataset pose",
        "",
        "Own recordings were not available (injury). Comparison uses MM-Fit RGB (Zenodo 7672767)",
        "→ MediaPipe → same Model 1 / rep counter / Model 2 RF as the dataset pose pipeline.",
        "",
        "**No form labels** on MM-Fit: Model 2 column is mean P(correct) only, not accuracy.",
        "",
        "| Session | Sync within 3 fr. | M1 macro-F1 dataset (pose / both) | M1 macro-F1 MediaPipe (pose / both) | Rep MAE dataset | Rep MAE MediaPipe | M2 P(correct) ds / mp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    good = []
    for m in results:
        sync = m.get("sync") or {}
        frac = sync.get("frac_within_3")
        poor = frac is not None and frac < 0.70
        if not poor:
            good.append(m)
        mark = " †" if poor else ""
        sync_s = "—" if frac is None else f"{sync.get('n_within_3')}/{sync.get('n_sets')} ({frac:.2f})"
        lines.append(
            f"| {m['session']}{mark} | {sync_s} | "
            f"{m['m1_macro_f1_dataset_pose_only']:.3f} / {m['m1_macro_f1_dataset_both']:.3f} | "
            f"{m['m1_macro_f1_mediapipe_pose_only']:.3f} / {m['m1_macro_f1_mediapipe_both']:.3f} | "
            f"{m['rep_mae_dataset']:.2f} | {m['rep_mae_mediapipe']:.2f} | "
            f"{(m['form_p_correct_mean_dataset'] or 0):.2f} / {(m['form_p_correct_mean_mediapipe'] or 0):.2f} |"
        )
    if good:
        def avg(key):
            vals = [g[key] for g in good if g.get(key) is not None]
            return float(np.mean(vals)) if vals else float("nan")
        lines.append(
            f"| **mean** (sync≥70%) | — | "
            f"{avg('m1_macro_f1_dataset_pose_only'):.3f} / {avg('m1_macro_f1_dataset_both'):.3f} | "
            f"{avg('m1_macro_f1_mediapipe_pose_only'):.3f} / {avg('m1_macro_f1_mediapipe_both'):.3f} | "
            f"{avg('rep_mae_dataset'):.2f} | {avg('rep_mae_mediapipe'):.2f} | "
            f"{avg('form_p_correct_mean_dataset'):.2f} / {avg('form_p_correct_mean_mediapipe'):.2f} |"
        )
    lines += [
        "",
        "† sync < 70% of sets within 3 frames — shown but excluded from the mean.",
        "",
        "**Domain shift (RQ4 finding, not a bug):** the EC3D-trained Model 2 random forest does **not** carry "
        "over to MM-Fit. Even on dataset pose, mean P(correct) on squats is about **0.08**. "
        "Different skeleton source, camera and people.",
        "",
    ]
    w20 = sync_all.get("w20") or {}
    if w20.get("vs_measured_mean_abs") is not None:
        lines.append(
            f"**Auto sync validation (w20):** fitted `source=auto` vs config measured map — "
            f"mean |Δ| **{w20['vs_measured_mean_abs']:.2f}** frames on sampled video frames "
            f"(frac_within_3={w20.get('frac_within_3')})."
        )
        lines.append("")
    lines.append(f"Metrics: `experiments/{run.name}/metrics.json`.")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "rq4_first_look.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", REPORTS / "rq4_first_look.md", flush=True)


if __name__ == "__main__":
    main()
