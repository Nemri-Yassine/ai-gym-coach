"""Rep counting inside one set from the normalized skeleton.

The signal is the first principal component of the 36 joint coordinates,
smoothed with a 9-frame moving average. Peaks (or valleys, whichever is more
regular) are the reps.
"""
import numpy as np
from scipy.signal import find_peaks

from awc.config import VIDEO_FPS

SMOOTH = 9
MIN_SECONDS = 0.6
PROMINENCE = 0.3


def movement_signal(pose) -> np.ndarray:
    """First principal component of (T, 18, 2), smoothed. Length stays T."""
    seq = np.asarray(pose, dtype=np.float64).reshape(len(pose), -1)
    seq = seq - seq.mean(axis=0, keepdims=True)
    if not np.isfinite(seq).all() or np.allclose(seq, 0):
        return np.zeros(len(pose), dtype=np.float64)
    # Economy SVD: the first right singular vector is the main movement direction.
    _, _, vt = np.linalg.svd(seq, full_matrices=False)
    signal = seq @ vt[0]
    kernel = np.ones(SMOOTH) / SMOOTH
    pad = SMOOTH // 2
    return np.convolve(np.pad(signal, pad, mode="edge"), kernel, mode="valid")


def _interval_var(peaks):
    """Lower is more regular. Fewer than two peaks cannot be scored."""
    if len(peaks) < 2:
        return np.inf
    if len(peaks) == 2:
        return 0.0
    return float(np.var(np.diff(peaks)))


def _peaks(signal, distance):
    std = float(np.std(signal))
    if std < 1e-8:
        return np.array([], dtype=int)
    idx, _ = find_peaks(signal, distance=distance, prominence=PROMINENCE * std)
    return idx


def count_reps(pose, fps=VIDEO_FPS):
    """Return (count, boundaries) for one set. boundaries are (start, end) index pairs."""
    signal = movement_signal(pose)
    distance = max(1, int(round(MIN_SECONDS * fps)))
    forward = _peaks(signal, distance)
    backward = _peaks(-signal, distance)
    var_f, var_b = _interval_var(forward), _interval_var(backward)
    if var_f < var_b or (var_f == var_b and len(forward) >= len(backward)):
        chosen, used = forward, signal
    else:
        chosen, used = backward, -signal
    if len(chosen) == 0:
        return 0, []
    valleys = [int(np.argmin(used[:chosen[0] + 1]))]
    for a, b in zip(chosen[:-1], chosen[1:]):
        valleys.append(int(a + np.argmin(used[a:b + 1])))
    valleys.append(int(chosen[-1] + np.argmin(used[chosen[-1]:])))
    boundaries = [(valleys[i], valleys[i + 1]) for i in range(len(chosen))]
    return int(len(chosen)), boundaries
