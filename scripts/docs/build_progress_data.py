"""Real data for the week 1 progress presentation (presentation/progress.html).

    python scripts/docs/build_progress_data.py

Writes presentation/progress_data.js (const PROGRESS = {...}). Everything shown in the simulations
comes from here: MM-Fit skeletons of an unseen person (w00 is in test_unseen), the trained
camera-only Model 1's real predictions, both rep counters run on the same sets, and one
public RepCount video with its real prediction.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from awc.config import INTERIM, MMFIT_CLASSES, MODELS, PRESENTATION
from awc.data import mmfit
from awc.features import reps as R
from awc.features.skeleton import normalize
from awc.features.sync import POSE_X, POSE_Y, pose_to_frame
from awc.models.m1_recognition import M1Model

WINDOW = 90
SESSION = "w00"                 # test_unseen: a person the model never saw
DISPLAY_STEP = 3                # 10 fps in the presentation
CLIP_SECONDS = 10


def r(a, n=2):
    return [round(float(v), n) for v in a]


def load_model(name):
    ckpt = torch.load(MODELS / f"{name}.pt", map_location="cpu", weights_only=False)
    model = M1Model(ckpt["arch"], ckpt["inputs"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def probs_for(model, pose, ends):
    """Softmax of Model 1 for the causal window [end-90, end) at every end frame."""
    xs, keep = [], []
    for e in ends:
        w = pose[e - WINDOW:e]
        if e - WINDOW < 0 or not np.isfinite(w).all():
            keep.append(False)
            continue
        xs.append(normalize(w).reshape(WINDOW, 36))
        keep.append(True)
    out = [None] * len(ends)
    if xs:
        with torch.no_grad():
            p = torch.softmax(model(torch.from_numpy(np.asarray(xs, np.float32))), 1).numpy()
        it = iter(p)
        out = [r(next(it), 3) if k else None for k in keep]
    return out


def skel(frames):
    """(T, 18, 2) with y up -> [[xs], [ys]] in screen orientation (y down), for the canvas player."""
    return [[r(f[:, 0], 2), r(-f[:, 1], 2)] for f in frames]


def synced_row(frame=4100):
    """One real row of the Spark output (data/interim/mmfit_synced), for the Spark slide."""
    import pandas as pd
    df = pd.read_parquet(INTERIM / "mmfit_synced" / f"session={SESSION}")
    row = df[df["frame"] == frame].iloc[0]
    keep = ["frame", "x0", "y0", "x8", "y8", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "hr", "hr_mask", "exercise", "set_id"]
    out = {}
    for k in keep:
        v = row[k]
        out[k] = v if isinstance(v, str) else (None if pd.isna(v) else round(float(v), 3))
    return out


def session_pose():
    df = pose_to_frame(mmfit.load_stream(SESSION, "pose_2d", mmap=False))
    pose = np.stack([df[POSE_X].to_numpy(), df[POSE_Y].to_numpy()], axis=-1)
    return df["frame"].to_numpy(), pose


def model1_clips(model, frames, pose):
    clips = {}
    for start, end, reps, ex in mmfit.load_labels(SESSION):
        if ex in clips:
            continue
        # start once a full 3 s window lies inside the set, so the model reads movement only
        i0 = int(np.searchsorted(frames, start)) + WINDOW
        ends = list(range(i0, min(i0 + CLIP_SECONDS * 30, int(np.searchsorted(frames, end))), DISPLAY_STEP))
        probs = probs_for(model, pose, ends)
        shown = pose[ends]
        clips[ex] = {"frames": skel(shown), "probs": probs, "reps": reps}
        valid = [p for p in probs if p]
        top = Counter(int(np.argmax(p)) for p in valid).most_common(1)[0][0] if valid else None
        clips[ex]["top"] = MMFIT_CLASSES[top] if top is not None else None
    return clips


def rep_compare(frames, pose, exercise):
    start, end, reps, _ = next(l for l in mmfit.load_labels(SESSION) if l[3] == exercise)
    m = (frames >= start) & (frames <= end)
    seq = normalize(pose[m])
    old_n, old_b = R.count_reps(seq)                         # first counter: PCA only
    new_n, new_b = R.count_reps(seq, exercise=exercise)      # fixed counter
    step = 2
    old_sig = R.movement_signal(seq)
    new_sig = R.legs_signal(seq) if R.SIGNAL_BY_EXERCISE.get(exercise) == "legs" else None
    if new_sig is None:
        right, left = R.arm_signals(seq)
        new_sig = right - left if R.SIGNAL_BY_EXERCISE.get(exercise) == "arms" else old_sig
    norm = lambda s: (s - s.mean()) / (s.std() + 1e-9)
    return {
        "exercise": exercise, "true": int(reps), "old": int(old_n), "new": int(new_n),
        "fps": 30 / step,
        "frames": skel(seq[::step]),
        "old_signal": r(norm(old_sig)[::step], 2), "new_signal": r(norm(new_sig)[::step], 2),
        "old_done": [int(b[1] // step) for b in old_b], "new_done": [int(b[1] // step) for b in new_b],
    }


def public_video(model):
    """One RepCount squat video, camera never seen in training, with the real prediction."""
    from awc.data import repcount
    items = {i["video_id"]: i for i in repcount.videos()}
    cache = INTERIM / "repcount" / "pose"
    for vid, item in items.items():
        if item["exercise"] != "squats" or not (cache / f"{vid}.npz").exists():
            continue
        z = np.load(cache / f"{vid}.npz")
        pose, fps = z["pose"], float(z["fps"])
        bounds = item["cycle_bounds"]
        s = int(np.floor(min(bounds) * 30 / fps)) if bounds else 0
        ends = list(range(max(s, WINDOW), min(s + CLIP_SECONDS * 30, len(pose)), DISPLAY_STEP))
        if len(ends) < 60:
            continue
        probs = probs_for(model, pose, ends)
        if sum(p is not None for p in probs) < 0.9 * len(probs):
            continue
        votes = Counter(int(np.argmax(p)) for p in probs if p)
        top = MMFIT_CLASSES[votes.most_common(1)[0][0]]
        if top != "squats":
            continue
        shown = np.nan_to_num(pose[ends], nan=0.0)
        return {"video_id": vid, "true": item["exercise"], "count": item["count"], "top": top,
                "frames": skel(shown), "probs": probs}
    return None


def main():
    torch.set_num_threads(4)
    model = load_model("m1_transformer_pose")
    frames, pose = session_pose()
    data = {
        "classes": MMFIT_CLASSES,
        "session": SESSION,
        "model1": model1_clips(model, frames, pose),
        "reps": {ex: rep_compare(frames, pose, ex) for ex in ("lunges", "bicep_curls", "squats")},
        "public": public_video(model),
        "synced_row": synced_row(),
    }
    out = PRESENTATION / "progress_data.js"
    out.write_text("const PROGRESS = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    for ex, c in data["model1"].items():
        print(f"  model1 {ex:24s} -> {c['top']}")
    for ex, c in data["reps"].items():
        print(f"  reps {ex}: true {c['true']} old {c['old']} new {c['new']}")
    p = data["public"]
    print("  public:", p and (p["video_id"], p["true"], p["top"], p["count"]))
    print("  synced row:", data["synced_row"])


if __name__ == "__main__":
    main()
