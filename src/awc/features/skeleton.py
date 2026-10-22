"""One 18-joint skeleton for MM-Fit, EC3D and MediaPipe.

Coordinates are (x right, y up). normalize() centers each frame on the mid-hip
and scales by the median neck-to-mid-hip distance.
"""
import numpy as np

from awc.data.mmfit import JOINTS_2D

# BODY_25 index 8 is mid-hip and is not part of COCO-18.
EC3D_TO_COCO18 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

# MediaPipe Pose landmark indices. Neck is the midpoint of the two shoulders.
MEDIAPIPE_TO_COCO18 = {
    "nose": 0,
    "neck": (11, 12),
    "r_shoulder": 12,
    "r_elbow": 14,
    "r_wrist": 16,
    "l_shoulder": 11,
    "l_elbow": 13,
    "l_wrist": 15,
    "r_hip": 24,
    "r_knee": 26,
    "r_ankle": 28,
    "l_hip": 23,
    "l_knee": 25,
    "l_ankle": 27,
    "r_eye": 5,
    "l_eye": 2,
    "r_ear": 8,
    "l_ear": 7,
}

# 17 bones, same pairs as the presentation (COCO-18 indices).
BONES = [
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
    (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
]

# Swap left/right joints, then negate x, to mirror a sequence.
_LR_SWAP = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]

R_HIP, L_HIP, NECK = 8, 11, 1


def mediapipe_to_coco18(landmarks, y_down=True) -> np.ndarray:
    """MediaPipe landmarks (33, 2+) in image coordinates → (18, 2) with y up."""
    lm = np.asarray(landmarks, dtype=np.float64)
    xy = lm[:, :2].copy()
    if y_down:
        xy[:, 1] *= -1
    out = np.empty((18, 2), dtype=np.float32)
    for i, name in enumerate(JOINTS_2D):
        spec = MEDIAPIPE_TO_COCO18[name]
        if name == "neck":
            a, b = spec
            out[i] = ((xy[a] + xy[b]) / 2).astype(np.float32)
        else:
            out[i] = xy[spec].astype(np.float32)
    return out


def normalize(seq) -> np.ndarray:
    """(T, 18, 2) → float32, origin at mid-hip, unit = median torso length.

    Invariant to translation and to a uniform scale.
    """
    seq = np.asarray(seq, dtype=np.float64)
    mid = seq[:, [R_HIP, L_HIP], :].mean(axis=1)
    centered = seq - mid[:, None, :]
    torso = np.linalg.norm(centered[:, NECK, :], axis=1)
    scale = float(np.median(torso))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return (centered / scale).astype(np.float32)


def _facing_yaw(p3d):
    """Yaw of the horizontal facing direction, toes minus heels (BODY_25)."""
    toe = (p3d[:, :2, 19] + p3d[:, :2, 22]) / 2
    heel = (p3d[:, :2, 21] + p3d[:, :2, 24]) / 2
    direction = (toe - heel).mean(axis=0)
    return float(np.arctan2(direction[1], direction[0]))


def project_ec3d(p3d, yaw_deg) -> np.ndarray:
    """EC3D (T, 3, 25), z up → (T, 18, 2) as seen from yaw_deg off the facing direction.

    0° looks at the person, 90° is a side view. Image x is horizontal, image y is z.
    """
    p3d = np.asarray(p3d, dtype=np.float64)
    angle = _facing_yaw(p3d) + np.radians(yaw_deg)
    right = np.array([-np.sin(angle), np.cos(angle)])
    horizontal = np.einsum("tcj,c->tj", p3d[:, :2, :], right)
    vertical = p3d[:, 2, :]
    idx = EC3D_TO_COCO18
    seq = np.stack([horizontal[:, idx], vertical[:, idx]], axis=-1)
    return seq.astype(np.float32)


def mirror(seq) -> np.ndarray:
    """Flip x and swap left/right joints. Applying it twice returns the original."""
    out = np.asarray(seq, dtype=np.float32)[:, _LR_SWAP, :].copy()
    out[:, :, 0] *= -1
    return out
