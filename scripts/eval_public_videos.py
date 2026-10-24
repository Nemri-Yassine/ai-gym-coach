"""Model 1 and rep counting on public gym videos the models have never seen (RepCount-A test split).

    python scripts/eval_public_videos.py --extract --workers 4
    python scripts/eval_public_videos.py

--extract runs MediaPipe on every usable video and caches the skeleton in
data/interim/repcount/pose/. Without it, only the cached skeletons are evaluated.
"""
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from awc.config import FIGURES, INTERIM, MMFIT_CLASSES, MODELS, REPORTS, VIDEO_FPS
from awc.data import repcount
from awc.evaluation.metrics import plot_confusion, rep_count_errors
from awc.features.reps import count_reps
from awc.features.skeleton import normalize
from awc.models.m1_recognition import M1Model
from awc.training.tracking import new_run, save_metrics

CACHE = INTERIM / "repcount"
POSE_DIR = CACHE / "pose"
VIDEO_DIR = CACHE / "videos"
CHECKPOINTS = ("m1_transformer_both", "m1_transformer_pose", "m1_cnn_both")
WINDOW, HOP = 90, 45
MIN_DETECTED = 0.5


def _extract_one(item):
    from awc.features.pose_video import extract_pose
    vid = item["video_id"]
    out = POSE_DIR / f"{vid}.npz"
    if out.exists():
        return vid, "cached"
    mp4 = VIDEO_DIR / f"{vid}.mp4"
    if not mp4.exists():
        repcount.write_video(item["row"], mp4)
    t0 = time.perf_counter()
    pose, fps, n_src = extract_pose(mp4)
    np.savez_compressed(out, pose=pose, fps=fps, n_src=n_src)
    mp4.unlink()
    return vid, f"{n_src} frames {time.perf_counter() - t0:.0f}s"


def extract(items, workers):
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    todo = [i for i in items if not (POSE_DIR / f"{i['video_id']}.npz").exists()]
    print(f"{len(items) - len(todo)} cached, {len(todo)} to extract", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_extract_one, i) for i in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            vid, msg = fut.result()
            print(f"[{k}/{len(todo)}] {vid} {msg}", flush=True)


def action_span(bounds, fps_src, n_frames):
    """First rep start to last rep end, in 30 fps frames."""
    if not bounds:
        return 0, n_frames
    scale = VIDEO_FPS / fps_src
    start = int(np.floor(min(bounds) * scale))
    end = int(np.ceil(max(bounds) * scale))
    return max(0, start), min(n_frames, end + 1)


def windows(pose, start, end):
    """Normalized 90-frame windows whose centre lies inside [start, end)."""
    out = []
    for s in range(0, len(pose) - WINDOW + 1, HOP):
        centre = s + WINDOW // 2
        if not start <= centre < end:
            continue
        w = pose[s:s + WINDOW]
        if np.isfinite(w).all():
            out.append(normalize(w).reshape(WINDOW, 36))
    return np.asarray(out, dtype=np.float32).reshape(-1, WINDOW, 36)


def load_model(name):
    ckpt = torch.load(MODELS / f"{name}.pt", map_location="cpu", weights_only=False)
    model = M1Model(ckpt["arch"], ckpt["inputs"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def predict(model, x):
    """Pose-only prediction. Models trained with IMU get a zero IMU and mask 0."""
    if len(x) == 0:
        return np.zeros(0, dtype=np.int64)
    pose = torch.from_numpy(x)
    with torch.no_grad():
        if model.inputs == "pose":
            logits = model(pose)
        else:
            logits = model(pose, torch.zeros(len(x), WINDOW, 6), torch.zeros(len(x)))
    return logits.argmax(1).numpy()


def video_vote(preds):
    """Most common exercise among the windows, ignoring non_activity when anything else was seen."""
    na = MMFIT_CLASSES.index("non_activity")
    active = [int(p) for p in preds if p != na]
    pool = active or [int(p) for p in preds]
    return Counter(pool).most_common(1)[0][0] if pool else na


def evaluate(items):
    models = {name: load_model(name) for name in CHECKPOINTS if (MODELS / f"{name}.pt").exists()}
    labels = list(range(len(MMFIT_CLASSES)))
    rec = {name: {"y": [], "p": [], "vy": [], "vp": []} for name in models}
    reps = []
    skipped = []
    for item in items:
        path = POSE_DIR / f"{item['video_id']}.npz"
        if not path.exists():
            skipped.append((item["video_id"], "not extracted"))
            continue
        z = np.load(path)
        pose, fps = z["pose"], float(z["fps"])
        start, end = action_span(item["cycle_bounds"], fps, len(pose))
        segment = pose[start:end]
        detected = float(np.isfinite(segment).all(axis=(1, 2)).mean()) if len(segment) else 0.0
        if detected < MIN_DETECTED:
            skipped.append((item["video_id"], f"person found in {detected:.0%} of frames"))
            continue
        finite = segment[np.isfinite(segment).all(axis=(1, 2))]
        # Exercise given as known (oracle), so this measures the counter, not Model 1.
        pred_reps, _ = count_reps(normalize(finite), exercise=item["exercise"] if item["mmfit"] else None)
        reps.append({"video_id": item["video_id"], "exercise": item["exercise"],
                     "true": item["count"], "pred": int(pred_reps)})
        if not item["mmfit"]:
            continue
        x = windows(pose, start, end)
        if len(x) == 0:
            continue
        y = MMFIT_CLASSES.index(item["exercise"])
        for name, model in models.items():
            p = predict(model, x)
            rec[name]["y"] += [y] * len(p)
            rec[name]["p"] += p.tolist()
            rec[name]["vy"].append(y)
            rec[name]["vp"].append(video_vote(p))

    m1 = {}
    for name, r in rec.items():
        y, p = np.array(r["y"]), np.array(r["p"])
        vy, vp = np.array(r["vy"]), np.array(r["vp"])
        per_ex = {}
        for ex in ("squats", "pushups", "situps", "jumping_jacks"):
            k = MMFIT_CLASSES.index(ex)
            m = vy == k
            per_ex[ex] = {"videos": int(m.sum()), "video_accuracy": float((vp[m] == k).mean()) if m.any() else None}
        predicted = Counter(MMFIT_CLASSES[int(v)] for v in p)
        m1[name] = {
            "windows": int(len(y)),
            "window_accuracy": float((y == p).mean()) if len(y) else None,
            "videos": int(len(vy)),
            "video_accuracy": float((vy == vp).mean()) if len(vy) else None,
            "per_exercise": per_ex,
            "window_predictions": dict(predicted.most_common()),
            "confusion": np.array(
                [[int(((y == a) & (p == b)).sum()) for b in labels] for a in labels]).tolist(),
        }

    by_ex = {}
    for r in reps:
        by_ex.setdefault(r["exercise"], ([], []))
        by_ex[r["exercise"]][0].append(r["true"])
        by_ex[r["exercise"]][1].append(r["pred"])
    rep_table = {ex: rep_count_errors(t, p) for ex, (t, p) in sorted(by_ex.items())}
    mmfit_ex = [ex for ex in rep_table if ex in repcount.TO_MMFIT.values()]
    return {
        "model1": m1,
        "rep_counting": {
            "per_exercise": rep_table,
            "macro_mae_mmfit_classes": float(np.mean([rep_table[e]["mae"] for e in mmfit_ex])) if mmfit_ex else None,
            "macro_mae_all": float(np.mean([v["mae"] for v in rep_table.values()])) if rep_table else None,
            "videos": reps,
        },
        "skipped": skipped,
        "n_items": len(items),
    }


def write_outputs(result, run_dir):
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    m1 = result["model1"]
    rc = result["rep_counting"]
    lines = [
        "# Public videos: Model 1 and rep counting on unseen cameras",
        "",
        "RepCount-A test split (TransRAC, CVPR 2022): YouTube gym videos with a rep count per video.",
        "Skeleton from MediaPipe Pose Landmarker (full), mapped to COCO-18, resampled to 30 fps.",
        "Pose only: these videos have no watch data. Models trained with IMU get a zero IMU and mask 0.",
        "Only the span from the first labelled rep to the last is used.",
        "",
        f"Videos considered: {result['n_items']}. Skipped: {len(result['skipped'])}.",
        "",
        "## Model 1 (squats, pushups, situps, jumping jacks)",
        "",
        "| Checkpoint | Videos | Video accuracy | Windows | Window accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in m1.items():
        va = "–" if r["video_accuracy"] is None else f"{r['video_accuracy'] * 100:.1f}%"
        wa = "–" if r["window_accuracy"] is None else f"{r['window_accuracy'] * 100:.1f}%"
        lines.append(f"| {name} | {r['videos']} | {va} | {r['windows']} | {wa} |")
    best = max(m1, key=lambda n: m1[n]["video_accuracy"] or 0) if m1 else None
    if best:
        lines += ["", f"Per exercise, `{best}` (video accuracy):", "",
                  "| Exercise | Videos | Video accuracy |", "|---|---:|---:|"]
        for ex, v in m1[best]["per_exercise"].items():
            acc = "–" if v["video_accuracy"] is None else f"{v['video_accuracy'] * 100:.1f}%"
            lines.append(f"| {ex} | {v['videos']} | {acc} |")
        top = ", ".join(f"{k} {v}" for k, v in list(m1[best]["window_predictions"].items())[:6])
        lines += ["", f"Most common window predictions: {top}."]
        cm = np.array(m1[best]["confusion"])
        plot_confusion(cm, MMFIT_CLASSES, run_dir / "confusion_public_videos.png",
                       title=f"{best} on RepCount-A (pose only)")
        plot_confusion(cm, MMFIT_CLASSES, FIGURES / "m1_confusion_public_videos.png",
                       title=f"{best} on RepCount-A (pose only)")
    lines += [
        "",
        "## Rep counting",
        "",
        f"Macro MAE on the four MM-Fit exercises: {rc['macro_mae_mmfit_classes']:.2f}. "
        f"Over every exercise: {rc['macro_mae_all']:.2f}.",
        "",
        "| Exercise | Videos | Exact | Within ±1 | MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for ex, s in rc["per_exercise"].items():
        lines.append(f"| {ex} | {s['n']} | {s['exact'] * 100:.1f}% | {s['within_1'] * 100:.1f}% | {s['mae']:.2f} |")
    if result["skipped"]:
        lines += ["", "Skipped videos:", ""] + [f"- `{v}`: {why}" for v, why in result["skipped"]]
    text = "\n".join(lines) + "\n"
    (REPORTS / "public_videos.md").write_text(text, encoding="utf-8")
    (run_dir / "public_videos.md").write_text(text, encoding="utf-8")

    names = list(rc["per_exercise"])
    maes = [rc["per_exercise"][n]["mae"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(range(len(names)), maes, color="#1F7F52")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_ylabel("MAE (reps)")
    ax.set_title("Rep counting on RepCount-A videos")
    fig.tight_layout()
    fig.savefig(FIGURES / "rep_counting_public_videos.png", dpi=120)
    fig.savefig(run_dir / "rep_counting_public_videos.png", dpi=120)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--extract", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="Only the first N usable videos")
    args = p.parse_args(argv)
    items = repcount.videos()
    if args.limit:
        items = items[:args.limit]
    if args.extract:
        extract(items, args.workers)
    result = evaluate(items)
    run_dir = new_run("public_videos", {"dataset": "RepCount-A test", "pose_model": "pose_landmarker_full",
                                        "checkpoints": list(CHECKPOINTS), "min_detected": MIN_DETECTED})
    save_metrics(run_dir, result)
    write_outputs(result, run_dir)
    for name, r in result["model1"].items():
        print(f"{name}: video acc {r['video_accuracy']}, window acc {r['window_accuracy']}")
    print(f"rep MAE (MM-Fit exercises) {result['rep_counting']['macro_mae_mmfit_classes']}")
    print(f"skipped {len(result['skipped'])}")
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
