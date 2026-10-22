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
