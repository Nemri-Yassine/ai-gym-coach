"""Rep counting inside one set from the normalized skeleton (origin at mid-hip, y up).

The signal depends on the exercise, which Model 1 provides:
- legs: mean ankle height relative to the hips. It shrinks at the bottom of every squat
  or lunge whichever leg is in front, so alternating lunges are not counted as half.
- arms: each wrist's height relative to its own shoulder. Arms moving together are
  counted once; alternating arms (MM-Fit bicep curls) are counted per arm and summed.
- pca: first principal component of the 36 coordinates, for the other exercises and
  when the exercise is unknown.
The choice per exercise was measured on the 616 MM-Fit sets (reports/rep_counting.md);
for jumping jacks, where MM-Fit was a tie, the RepCount public videos decided.
Every signal is smoothed with a 9-frame moving average; peaks or valleys, whichever
are more regular, are the reps.
"""
import numpy as np
from scipy.signal import find_peaks

from awc.config import VIDEO_FPS

SMOOTH = 9
MIN_SECONDS = 0.6
PROMINENCE = 0.3
ARMS_TOGETHER_CORR = 0.5

R_SHOULDER, R_WRIST, L_SHOULDER, L_WRIST = 2, 4, 5, 7
R_ANKLE, L_ANKLE = 10, 13

SIGNAL_BY_EXERCISE = {
    "squats": "legs",
    "lunges": "legs",
    "pushups": "legs",
    "bicep_curls": "arms",
    "dumbbell_rows": "arms",
    "lateral_shoulder_raises": "arms",
    "jumping_jacks": "pca",   # tied with legs on MM-Fit (0.44 vs 0.42); pca holds up on RepCount videos
    "dumbbell_shoulder_press": "pca",
    "situps": "pca",
    "tricep_extensions": "pca",
}


def _smooth(signal) -> np.ndarray:
    kernel = np.ones(SMOOTH) / SMOOTH
    pad = SMOOTH // 2
    return np.convolve(np.pad(signal, pad, mode="edge"), kernel, mode="valid")


def movement_signal(pose) -> np.ndarray:
    """First principal component of (T, 18, 2), smoothed. Length stays T."""
    seq = np.asarray(pose, dtype=np.float64).reshape(len(pose), -1)
    seq = seq - seq.mean(axis=0, keepdims=True)
    if not np.isfinite(seq).all() or np.allclose(seq, 0):
        return np.zeros(len(pose), dtype=np.float64)
    # Economy SVD: the first right singular vector is the main movement direction.
    _, _, vt = np.linalg.svd(seq, full_matrices=False)
    return _smooth(seq @ vt[0])


def legs_signal(pose) -> np.ndarray:
    """Mean ankle height relative to the hips, smoothed."""
    pose = np.asarray(pose, dtype=np.float64)
    return _smooth((pose[:, R_ANKLE, 1] + pose[:, L_ANKLE, 1]) / 2)


def arm_signals(pose):
    """(right, left) wrist height relative to the same-side shoulder, smoothed."""
    pose = np.asarray(pose, dtype=np.float64)
    right = _smooth(pose[:, R_WRIST, 1] - pose[:, R_SHOULDER, 1])
    left = _smooth(pose[:, L_WRIST, 1] - pose[:, L_SHOULDER, 1])
    return right, left


def _interval_var(peaks):
    """Lower is more regular. Fewer than two peaks cannot be scored."""
    if len(peaks) < 2:
        return np.inf
    if len(peaks) == 2:
        return 0.0
    return float(np.var(np.diff(peaks)))


def _peaks(signal, distance):
    std = float(np.std(signal))
    if std < 1e-8 or not np.isfinite(std):
        return np.array([], dtype=int)
    idx, _ = find_peaks(signal, distance=distance, prominence=PROMINENCE * std)
    return idx


def _count_on(signal, fps):
    """Count reps on one smoothed signal. Returns (count, boundaries)."""
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


def _count_arms(pose, fps):
    right, left = arm_signals(pose)
    together = np.std(right) > 1e-8 and np.std(left) > 1e-8 and np.corrcoef(right, left)[0, 1] > ARMS_TOGETHER_CORR
    if together:
        return _count_on((right + left) / 2, fps)
    n_r, b_r = _count_on(right, fps)
    n_l, b_l = _count_on(left, fps)
    return n_r + n_l, sorted(b_r + b_l)


def count_reps(pose, fps=VIDEO_FPS, exercise=None):
    """Return (count, boundaries) for one set. boundaries are (start, end) index pairs.

    exercise is an MM-Fit class name; None (or an unknown name) uses the PCA signal.
    """
    method = SIGNAL_BY_EXERCISE.get(exercise, "pca")
    if method == "legs":
        return _count_on(legs_signal(pose), fps)
    if method == "arms":
        return _count_arms(pose, fps)
    return _count_on(movement_signal(pose), fps)
