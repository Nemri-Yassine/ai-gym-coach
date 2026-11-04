"""Hand-made form measures for one rep.

The 3D measures are the ones used by the EC3D analysis (BODY_25, z up).
The 2D measures are the same six ideas on the projected phone skeleton
(COCO-18, x right, y up). That skeleton has no toes, so knee_toe uses the ankle.
"""
import numpy as np

FEATURE_NAMES = ("knee_width", "feet_width", "depth", "lean", "knee_toe", "sag")


def _dist3(p, a, b):
    return np.linalg.norm(p[:, :, a] - p[:, :, b], axis=1)


def features_3d(p) -> dict:
    """One number per known mistake. p is (T, 3, 25), z up, mid-hip at the origin."""
    hipw, shw = _dist3(p, 9, 12), _dist3(p, 2, 5)
    ank = (p[:, 2, 11] + p[:, 2, 14]) / 2
    leg = _dist3(p, 9, 11).mean() + 1e-9
    # EC3D keeps the trunk upright (neck fixed above mid-hip), so measure the trunk angle against the shins
    trunk = p[:, :, 1] - p[:, :, 8]
    shin = (p[:, :, 10] + p[:, :, 13]) / 2 - (p[:, :, 11] + p[:, :, 14]) / 2
    cos = np.einsum("ij,ij->i", trunk, shin) / (np.linalg.norm(trunk, axis=1) * np.linalg.norm(shin, axis=1) + 1e-9)
    lean = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    # body line for planks: distance of mid-hip from the shoulder-ankle line, signed by vertical side
    sh = (p[:, :, 2] + p[:, :, 5]) / 2
    an = (p[:, :, 11] + p[:, :, 14]) / 2
    t = np.clip(
        np.einsum("ij,ij->i", -sh, an - sh) / (np.einsum("ij,ij->i", an - sh, an - sh) + 1e-9),
        0, 1,
    )
    proj = sh + t[:, None] * (an - sh)
    sag = -proj[:, 2] / (_dist3(p, 2, 11).mean() + 1e-9)
    kt = []
    for kn, toe in ((10, 22), (13, 19)):
        kt.append(np.linalg.norm(p[:, :2, kn], axis=1) - np.linalg.norm(p[:, :2, toe], axis=1))
    return dict(
        knee_width=float(((_dist3(p, 10, 13) / hipw).min())),
        feet_width=float(((_dist3(p, 11, 14) / shw).mean())),
        depth=float((-ank).min() / leg),
        lean=float(lean.max()),
        knee_toe=float(np.max(kt) / leg),
        sag=float(np.median(sag)),
    )


def _dist2(p, a, b):
    return np.linalg.norm(p[:, a, :] - p[:, b, :], axis=1)


def features_2d(p) -> dict:
    """Same six measures on a projected skeleton (T, 18, 2), x right, y up."""
    p = np.asarray(p, dtype=np.float64)
    mid = (p[:, 8, :] + p[:, 11, :]) / 2
    hipw = _dist2(p, 8, 11) + 1e-9
    shw = _dist2(p, 2, 5) + 1e-9
    leg = _dist2(p, 8, 10).mean() + 1e-9
    ank_y = (p[:, 10, 1] + p[:, 13, 1]) / 2
    trunk = p[:, 1, :] - mid
    shin = (p[:, 9, :] + p[:, 12, :]) / 2 - (p[:, 10, :] + p[:, 13, :]) / 2
    cos = np.einsum("ij,ij->i", trunk, shin) / (
        np.linalg.norm(trunk, axis=1) * np.linalg.norm(shin, axis=1) + 1e-9
    )
    lean = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    sh = (p[:, 2, :] + p[:, 5, :]) / 2 - mid
    an = (p[:, 10, :] + p[:, 13, :]) / 2 - mid
    t = np.clip(
        np.einsum("ij,ij->i", -sh, an - sh) / (np.einsum("ij,ij->i", an - sh, an - sh) + 1e-9),
        0, 1,
    )
    proj = sh + t[:, None] * (an - sh)
    sag = -proj[:, 1] / (_dist2(p, 2, 10).mean() + 1e-9)
    kt = []
    for kn, ank in ((9, 10), (12, 13)):
        kt.append(np.abs(p[:, kn, 0] - mid[:, 0]) - np.abs(p[:, ank, 0] - mid[:, 0]))
    height = mid[:, 1] - ank_y
    return dict(
        knee_width=float((_dist2(p, 9, 12) / hipw).min()),
        feet_width=float((_dist2(p, 10, 13) / shw).mean()),
        depth=float(height.min() / leg),
        lean=float(lean.max()),
        knee_toe=float(np.max(kt) / leg),
        sag=float(np.median(sag)),
    )


def feature_vector(measures) -> np.ndarray:
    return np.array([float(measures[name]) for name in FEATURE_NAMES], dtype=np.float64)
