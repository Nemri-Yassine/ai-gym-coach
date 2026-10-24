"""Rep counter on a sine-wave skeleton, and on the squat sets of session w00."""
import numpy as np
import pytest

from awc.data import mmfit
from awc.features.reps import count_reps
from awc.features.skeleton import normalize
from awc.features.sync import POSE_X, POSE_Y, pose_to_frame


def test_rep_count_sine():
    t = np.linspace(0, 10 * 2 * np.pi, 300, endpoint=False)
    pose = np.zeros((300, 18, 2), dtype=np.float64)
    pose[:, 8, 1] = np.sin(t)
    pose[:, 1, 1] = 1.0
    count, boundaries = count_reps(pose)
    assert count == 10
    assert len(boundaries) == 10


@pytest.mark.data
def test_rep_count_mmfit_squats():
    pose = pose_to_frame(mmfit.load_stream("w00", "pose_2d", mmap=False))
    errors = []
    for start, end, reps, exercise in mmfit.load_labels("w00"):
        if exercise != "squats":
            continue
        sub = pose[(pose["frame"] >= start) & (pose["frame"] <= end)]
        seq = np.stack([
            sub[POSE_X].to_numpy(dtype=np.float64),
            sub[POSE_Y].to_numpy(dtype=np.float64),
        ], axis=-1)
        pred, _ = count_reps(normalize(seq))
        errors.append(abs(pred - reps))
    assert errors, "w00 has no squat sets"
    assert float(np.mean(errors)) <= 1


def _alternating_lunges(n_reps=10, fps=30):
    """Hips drop once per rep, the front leg alternates: the old PCA counter saw half."""
    t = np.arange(n_reps * fps * 2) / fps
    pose = np.zeros((len(t), 18, 2), dtype=np.float64)
    pose[:, 1, 1] = 1.0                                   # neck above the hips
    depth = 0.5 * (1 - np.cos(2 * np.pi * t / 2))         # one dip every 2 s
    side = np.sign(np.sin(np.pi * t / 2))                 # front leg switches each rep
    pose[:, 10] = np.stack([0.3 * side, -2 + depth], -1)  # r_ankle
    pose[:, 13] = np.stack([-0.3 * side, -2 + depth], -1) # l_ankle
    return pose


def test_lunges_not_halved():
    count, _ = count_reps(_alternating_lunges(10), exercise="lunges")
    assert count == 10


def test_alternating_curls_counted_per_arm():
    fps, n = 30, 10                                       # 10 curls, 5 per arm, alternating
    t = np.arange(n * fps * 2) / fps
    pose = np.zeros((len(t), 18, 2), dtype=np.float64)
    pose[:, 1, 1] = 1.0
    pose[:, 2] = (-0.5, 1.0)                              # r_shoulder
    pose[:, 5] = (0.5, 1.0)                               # l_shoulder
    right = np.clip(np.sin(np.pi * t / 2), 0, None)       # right arm curls on odd seconds pairs
    left = np.clip(-np.sin(np.pi * t / 2), 0, None)       # left arm on the others
    pose[:, 4] = np.stack([np.full_like(t, -0.5), 0.2 + right], -1)
    pose[:, 7] = np.stack([np.full_like(t, 0.5), 0.2 + left], -1)
    count, boundaries = count_reps(pose, exercise="bicep_curls")
    assert count == n
    assert boundaries == sorted(boundaries)


def test_unknown_exercise_falls_back_to_pca():
    t = np.linspace(0, 10 * 2 * np.pi, 300, endpoint=False)
    pose = np.zeros((300, 18, 2), dtype=np.float64)
    pose[:, 8, 1] = np.sin(t)
    pose[:, 1, 1] = 1.0
    assert count_reps(pose, exercise="not_an_exercise")[0] == 10


@pytest.mark.data
def test_mmfit_lunges_and_curls_not_halved():
    pose = pose_to_frame(mmfit.load_stream("w00", "pose_2d", mmap=False))
    errors = []
    for start, end, reps, exercise in mmfit.load_labels("w00"):
        if exercise not in ("lunges", "bicep_curls"):
            continue
        sub = pose[(pose["frame"] >= start) & (pose["frame"] <= end)]
        seq = np.stack([sub[POSE_X].to_numpy(), sub[POSE_Y].to_numpy()], axis=-1)
        pred, _ = count_reps(normalize(seq), exercise=exercise)
        errors.append(abs(pred - reps))
    assert errors and float(np.mean(errors)) <= 1
