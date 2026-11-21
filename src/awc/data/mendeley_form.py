"""Mendeley multi-view good/bad form dataset → fixed-length 2D squat reps for Model 2.

Source: https://data.mendeley.com/datasets/kgbb3yn47p/2 (CC BY 4.0). 26 people in a real
gym, one good and one bad set per exercise, filmed front / side / diagonal at 15 fps.
Skeletons come from scripts/analysis/inspect_mendeley_form.py (MediaPipe → COCO-18,
y up, resampled to 30 fps) and are cached in data/interim/mendeley_form/pose/.

Labels are per video: every rep of a "bad" video is labelled bad. There is no mistake
type, so this data only trains the binary "is this rep correct?" question.
"""
import re

import numpy as np
import pandas as pd

from awc.config import DATA, VIDEO_FPS
from awc.features.form_dataset import REP_FRAMES, resample
from awc.features.reps import count_reps
from awc.features.skeleton import normalize

INTERIM = DATA / "interim" / "mendeley_form"
POSE_DIR = INTERIM / "pose"
REPS_NPZ = DATA / "processed" / "m2_mendeley_reps.npz"
NAME = re.compile(r"subject_(\d+)_(.+)_(good|bad)_(front|side|diagonal)$")

MIN_REP_S, MAX_REP_S = 0.8, 6.0
# Mendeley folder → MM-Fit class used by the rep counter.
REP_EXERCISE = {"squat": "squats", "push_up": "pushups"}


def video_reps(pose, exercise="squat", fps=VIDEO_FPS):
    """(T, 18, 2) skeleton of one set → list of (64, 18, 2) normalized reps."""
    finite = np.isfinite(pose).all(axis=(1, 2))
    if finite.sum() < fps:
        return []
    pose_n = normalize(pose[finite])
    _n, bounds = count_reps(pose_n, fps=fps, exercise=REP_EXERCISE[exercise])
    reps = []
    for a, b in bounds:
        length = b - a + 1
        if not (MIN_REP_S * fps <= length <= MAX_REP_S * fps):
            continue
        reps.append(resample(pose_n[a:b + 1], REP_FRAMES))
    return reps


def build_reps(exercise="squat", path=REPS_NPZ):
    """Cut every cached skeleton of one exercise into reps. Writes and returns the arrays."""
    rows, poses = [], []
    for npy in sorted(POSE_DIR.glob(f"subject_*_{exercise}_*.npy")):
        m = NAME.match(npy.stem)
        if not m or m.group(2) != exercise:
            continue
        subject, _ex, quality, view = m.groups()
        for i, rep in enumerate(video_reps(np.load(npy), exercise)):
            poses.append(rep)
            rows.append(dict(subject=f"m{subject}", quality=quality, view=view, rep=i, video=npy.stem))
    meta = pd.DataFrame(rows)
    pack = {
        "poses": np.stack(poses).astype(np.float32),
        "subject": meta.subject.to_numpy(str),
        "view": meta.view.to_numpy(str),
        "video": meta.video.to_numpy(str),
        "rep": meta.rep.to_numpy(np.int64),
        "is_correct": (meta.quality == "good").to_numpy(np.int64),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pack)
    print(f"mendeley {exercise} reps {len(meta)} from {meta.video.nunique()} videos, "
          f"{meta.subject.nunique()} people; good {int(pack['is_correct'].sum())} / bad "
          f"{int((1 - pack['is_correct']).sum())} -> {path}")
    return pack


def load_reps(path=REPS_NPZ):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


if __name__ == "__main__":
    build_reps()
