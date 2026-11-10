"""Diagnose systematic M1 failures on own videos vs MM-Fit train.

    python scripts/diagnose_own_m1.py

Writes reports/figures/own_m1_diagnosis.png — mean key-joint trajectories for
failed exercises: our sets vs MM-Fit train (true class) vs MM-Fit train (predicted).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from awc.config import FIGURES, MMFIT_SPLIT, OWN_INTERIM, REPORTS
from awc.data import mmfit
from awc.data.own_manifest import load_manifest
from awc.features.skeleton import normalize
from awc.features.sync import pose_to_frame

# (truth, typical predicted label on our camera)
FAILURES = [
    ("tricep_extensions", "non_activity"),
    ("lateral_shoulder_raises", "jumping_jacks"),
    ("situps", "pushups"),
    ("dumbbell_rows", "pushups"),
]

# COCO-18: L/R wrist, elbow, shoulder, ankle
KEY_JOINTS = {
    "L_wrist": 7, "R_wrist": 4,
    "L_elbow": 6, "R_elbow": 3,
    "L_shoulder": 5, "R_shoulder": 2,
    "L_ankle": 13, "R_ankle": 10,
}
POSE_X = [f"x{i}" for i in range(18)]
POSE_Y = [f"y{i}" for i in range(18)]
T_RESAMPLE = 64


def _resample_traj(pose_n: np.ndarray, t_out: int = T_RESAMPLE) -> np.ndarray:
    """(T, 18, 2) → (t_out, 18, 2) linear in time."""
    t = len(pose_n)
    if t < 4:
        return np.full((t_out, 18, 2), np.nan)
    idx = np.linspace(0, t - 1, t_out)
    out = np.empty((t_out, 18, 2), dtype=np.float64)
    base = np.arange(t)
    for j in range(18):
        for c in range(2):
            out[:, j, c] = np.interp(idx, base, pose_n[:, j, c])
    return out


def _mean_own(exercise: str) -> np.ndarray | None:
    rows = [r for r in load_manifest() if r["use"] == "eval" and r["exercise"] == exercise]
    trajs = []
    for r in rows:
        path = OWN_INTERIM / r["person"] / f"{r['session_folder']}_set{int(r['set']):02d}.parquet"
        if not path.is_file():
            continue
        df = pd.read_parquet(path)
        pose = np.stack([df[POSE_X].to_numpy(), df[POSE_Y].to_numpy()], axis=-1)
        good = np.isfinite(pose).all(axis=(1, 2))
        if good.sum() < 16:
            continue
        trajs.append(_resample_traj(normalize(pose[good])))
    if not trajs:
        return None
    return np.nanmean(np.stack(trajs), axis=0)


def _mean_mmfit_train(exercise: str, max_sets: int = 40) -> np.ndarray | None:
    """Average normalized pose over MM-Fit train sessions for this exercise label."""
    trajs = []
    for sid in MMFIT_SPLIT["train"]:
        if len(trajs) >= max_sets:
            break
        raw = mmfit.load_stream(sid, "pose_2d", mmap=False)
        if raw is None:
            continue
        try:
            df = pose_to_frame(raw)
        except ValueError:
            continue
        pose = np.stack(
            [df[[f"x{i}" for i in range(18)]].to_numpy(),
             df[[f"y{i}" for i in range(18)]].to_numpy()],
            axis=-1,
        )
        frames = df["frame"].to_numpy()
        for start, end, _reps, ex in mmfit.load_labels(sid):
            if ex != exercise:
                continue
            m = (frames >= start) & (frames <= end)
            seg = pose[m]
            if len(seg) < 16:
                continue
            good = np.isfinite(seg).all(axis=(1, 2))
            if good.sum() < 16:
                continue
            trajs.append(_resample_traj(normalize(seg[good])))
            if len(trajs) >= max_sets:
                break
    if not trajs:
        return None
    return np.nanmean(np.stack(trajs), axis=0)


def _series(traj: np.ndarray, joint: str, coord: str = "y") -> np.ndarray:
    j = KEY_JOINTS[joint]
    c = 0 if coord == "x" else 1
    return traj[:, j, c]


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(FAILURES), 3, figsize=(12, 3.2 * len(FAILURES)), sharey="row")
    if len(FAILURES) == 1:
        axes = np.array([axes])
    col_titles = ["Our videos (mean)", "MM-Fit train · true", "MM-Fit train · predicted"]
    for r, (truth, pred) in enumerate(FAILURES):
        own = _mean_own(truth)
        mm_true = _mean_mmfit_train(truth)
        mm_pred = None if pred == "non_activity" else _mean_mmfit_train(pred)
        panels = [own, mm_true, mm_pred]
        for c, traj in enumerate(panels):
            ax = axes[r, c]
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)
            if traj is None:
                ax.text(0.5, 0.5, "n/a" if pred != "non_activity" or c < 2 else "pred = non_activity",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9)
                ax.set_xticks([])
                continue
            t = np.linspace(0, 1, len(traj))
            for name, style in (("L_wrist", "-"), ("R_wrist", "--"),
                                ("L_ankle", ":"), ("R_ankle", "-.")):
                ax.plot(t, _series(traj, name, "y"), style, lw=1.2, label=name if r == 0 and c == 0 else None)
            ax.set_xlim(0, 1)
            ax.axhline(0, color="#999", lw=0.5)
            if c == 0:
                ax.set_ylabel(f"{truth}\n→ {pred}\n(y up)", fontsize=8)
            ax.set_xlabel("rep fraction" if r == len(FAILURES) - 1 else "")
        axes[r, 0].legend(loc="upper right", fontsize=7, ncol=2)
    fig.suptitle("Own M1 failures: mean key-joint Y trajectories (normalized, mid-hip origin)", fontsize=11)
    fig.tight_layout()
    out = FIGURES / "own_m1_diagnosis.png"
    fig.savefig(out, dpi=140)
    fig.savefig(REPORTS / "figures" / "own_m1_diagnosis.png", dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
