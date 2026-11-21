"""Inventory of the Mendeley multi-view good/bad form dataset (data/raw/mendeley-form).

1. Metadata for every video (fps, frames, duration, resolution) -> data/interim/mendeley_form/inventory.csv
2. Pose + rep count on a sample of squat and push-up videos -> data/interim/mendeley_form/pose_sample.csv
   (pose skeletons cached as .npy next to it).

Run: C:/awc_tmp/venv/Scripts/python.exe scripts/analysis/inspect_mendeley_form.py [--subjects 5]
"""
import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from awc.config import ROOT
from awc.features.pose_video import extract_pose
from awc.features.reps import count_reps
from awc.features.skeleton import normalize

RAW = ROOT / "data" / "raw" / "mendeley-form" / "videos"
OUT = ROOT / "data" / "interim" / "mendeley_form"
NAME = re.compile(r"subject_(\d+)_(.+)_(good|bad)_(front|side|diagonal)\.mp4$")
# Mendeley folder name -> MM-Fit class used by the rep counter. "bicep" is one stray file of bicep_curl.
# abs (dumbbell side bend) has no MM-Fit class: None -> PCA signal.
EXERCISE_MAP = {"squat": "squats", "push_up": "pushups", "bicep_curl": "bicep_curls", "bicep": "bicep_curls",
                "back": "dumbbell_rows", "shoulder": "dumbbell_shoulder_press",
                "tricep": "tricep_extensions", "abs": None}


def inventory():
    import cv2

    rows = []
    for path in sorted(RAW.rglob("*.mp4")):
        m = NAME.search(path.name)
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        rows.append(dict(file=path.relative_to(RAW).as_posix(),
                         subject=int(m.group(1)) if m else None,
                         exercise=m.group(2) if m else path.parent.parent.name,
                         quality=m.group(3) if m else path.parent.name,
                         view=m.group(4) if m else None,
                         fps=round(fps, 2), frames=n, seconds=round(n / fps, 2) if fps else None,
                         width=w, height=h, mb=round(path.stat().st_size / 1e6, 2)))
    return pd.DataFrame(rows)


def _pose_one(file):
    """Worker: extract (or load cached) skeleton + QA for one video. Runs in its own process."""
    stem = file.split("/")[-1][:-4]
    npy, qa_json = OUT / "pose" / f"{stem}.npy", OUT / "pose" / f"{stem}.qa.json"
    if npy.exists() and qa_json.exists():
        return file, np.load(npy), json.loads(qa_json.read_text())
    pose, _fps, _n, qa = extract_pose(RAW / file, return_qa=True)
    np.save(npy, pose)
    qa_json.write_text(json.dumps(qa))
    return file, pose, qa


def pose_sample(inv, n_subjects, workers):
    sample = inv[inv.exercise.isin(EXERCISE_MAP) & inv.subject.notna()]
    keep = sorted(sample.subject.unique())[:n_subjects]
    sample = sample[sample.subject.isin(keep)].set_index("file")
    (OUT / "pose").mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (file, pose, qa) in enumerate(pool.map(_pose_one, sample.index), 1):
            r = sample.loc[file]
            n_reps, _ = count_reps(normalize(pose), exercise=EXERCISE_MAP[r.exercise])
            rows.append(dict(file=file, subject=r.subject, exercise=r.exercise, quality=r.quality, view=r.view,
                             seconds=r.seconds, reps=n_reps, pct_detected=round(qa["pct_detected"], 3),
                             mean_visibility=round(qa["mean_visibility"], 3)))
            print(f"[{i}/{len(sample)}] {file}: {n_reps} reps, detected {qa['pct_detected']:.0%}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inv_path = OUT / "inventory.csv"
    inv = pd.read_csv(inv_path) if inv_path.exists() else inventory()
    inv.to_csv(inv_path, index=False)
    print(inv.groupby(["exercise", "quality"]).agg(videos=("file", "size"), subjects=("subject", "nunique"),
                                                   sec_median=("seconds", "median"), fps=("fps", "median")))
    print(inv.groupby("view").size(), inv[["width", "height"]].value_counts().head())
    ps = pose_sample(inv, args.subjects, args.workers)
    ps.to_csv(OUT / "pose_sample.csv", index=False)
    pd.set_option("display.max_rows", 200)
    print(ps.groupby(["exercise", "quality", "view"])[["reps", "pct_detected", "mean_visibility"]].median())
    # Same set filmed from 3 angles: how far apart are the counts? (a free consistency check)
    spread = ps.groupby(["exercise", "subject", "quality"]).reps.agg(lambda s: s.max() - s.min())
    print("cross-view rep spread (median, % sets within 2):")
    print(spread.groupby("exercise").agg(["median", lambda s: round((s <= 2).mean() * 100)]))


if __name__ == "__main__":
    main()
