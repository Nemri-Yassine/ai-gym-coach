"""Canonical skeleton: normalization, joint maps, EC3D projection, mirror."""
import numpy as np

from awc.data.mmfit import JOINTS_2D
from awc.features.skeleton import (
    BONES,
    EC3D_TO_COCO18,
    MEDIAPIPE_TO_COCO18,
    mediapipe_to_coco18,
    mirror,
    normalize,
    project_ec3d,
)


def test_normalize_invariance():
    rng = np.random.default_rng(0)
    seq = rng.normal(size=(20, 18, 2))
    seq[:, 1, 1] += 2.0  # a real torso, so the scale is not ~0
    shifted = (seq + np.array([3.0, -2.0])) * 2.5
    np.testing.assert_allclose(normalize(seq), normalize(shifted), atol=1e-5)


def test_mapping_lengths():
    assert len(EC3D_TO_COCO18) == 18
    assert len(MEDIAPIPE_TO_COCO18) == 18
    assert len(BONES) == 17
    assert list(MEDIAPIPE_TO_COCO18) == JOINTS_2D
    lm = np.zeros((33, 2))
    lm[11] = [2.0, 4.0]
    lm[12] = [6.0, 10.0]
    out = mediapipe_to_coco18(lm, y_down=True)
    assert out.shape == (18, 2)
    np.testing.assert_allclose(out[1], (out[2] + out[5]) / 2)


def test_project_front_vs_side():
    p = np.zeros((2, 3, 25), dtype=np.float64)
    p[:, 0, 19] = p[:, 0, 22] = 0.3
    p[:, 0, 21] = p[:, 0, 24] = -0.1
    p[:, 1, 2] = 0.4
    p[:, 1, 5] = -0.4
    p[:, 2, 1] = p[:, 2, 2] = p[:, 2, 5] = 1.0
    front = project_ec3d(p, 0)
    side = project_ec3d(p, 90)
    front_width = abs(float(front[0, 2, 0] - front[0, 5, 0]))
    side_width = abs(float(side[0, 2, 0] - side[0, 5, 0]))
    assert front_width > 0.5
    assert side_width < 1e-5


def test_mirror_twice_identity():
    rng = np.random.default_rng(1)
    seq = rng.normal(size=(5, 18, 2)).astype(np.float32)
    np.testing.assert_allclose(mirror(mirror(seq)), seq, atol=1e-6)
