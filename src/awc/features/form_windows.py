"""Model 2 on fixed time windows instead of reps: works for every exercise, no rep counter needed.

A bad set is bad all the way through (Mendeley labels are per video), so every 3-second
window of a set carries the set's label. Windows keep real time (no resampling to 64
frames), so speed and tempo stay visible to the model.

Input everywhere: one set's skeleton, (T, 18, 2) COCO-18, y up, 30 fps, already
`normalize`d (mid-hip origin, unit = median torso length).
"""
import numpy as np
import torch

from awc.config import VIDEO_FPS
from awc.features.form_features import FEATURE_NAMES, feature_vector, features_2d
from awc.features.reps import movement_signal
from awc.models.m2_form import joint_angles

WINDOW_S = 3.0
HOP_S = 1.5
WINDOW = int(WINDOW_S * VIDEO_FPS)   # 90 frames
HOP = int(HOP_S * VIDEO_FPS)         # 45 frames

# Mendeley folder -> our exercise name (MM-Fit class where one exists).
MENDELEY_EXERCISE = {
    "squat": "squats", "push_up": "pushups", "bicep_curl": "bicep_curls", "bicep": "bicep_curls",
    "back": "dumbbell_rows", "shoulder": "dumbbell_shoulder_press", "tricep": "tricep_extensions",
    "abs": "side_bends",
}
FORM_EXERCISES = ("squats", "pushups", "bicep_curls", "dumbbell_rows", "dumbbell_shoulder_press",
                  "tricep_extensions", "side_bends")

# Left/right angle pairs inside joint_angles: knees, hips, elbows, shoulders. Index 8 = trunk lean.
_LR_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))
# Bones for the body-proportion shortcut control (camera distance / angle / body shape, not form).
_PROPORTION_BONES = ((2, 5), (8, 11), (8, 9), (9, 10), (2, 3), (3, 4), (1, 8), (0, 1))


def active_span(pose, fps=VIDEO_FPS):
    """Trim the still start and end of a set (same rule as scripts/eval_own_recordings.activity_span)."""
    n = len(pose)
    if n < fps:
        return 0, n
    sig = movement_signal(pose)
    energy = np.abs(np.diff(sig, prepend=sig[0]))
    active = np.where(energy > float(np.median(energy) + 0.35 * np.std(energy)))[0]
    if len(active) < 5:
        trim = int(2 * fps)
        return trim, max(trim + 1, n - trim)
    max_trim = int(3 * fps)
    start = min(int(active[0]), max_trim)
    end = max(int(active[-1]) + 1, n - max_trim)
    return max(0, start), min(n, max(start + 1, end))


def set_windows(pose_n, trim=True):
    """(T, 18, 2) normalized set → (N, WINDOW, 18, 2) windows, hop 1.5 s. Windows with NaN are dropped."""
    pose_n = np.asarray(pose_n, dtype=np.float32)
    a, b = active_span(pose_n) if trim else (0, len(pose_n))
    seg = pose_n[a:b]
    out = [seg[s:s + WINDOW] for s in range(0, len(seg) - WINDOW + 1, HOP)]
    out = [w for w in out if np.isfinite(w).all()]
    if not out:
        return np.zeros((0, WINDOW, 18, 2), np.float32)
    return np.stack(out)


def _tempo_seconds(w):
    """Dominant period of the main movement inside the window (autocorrelation peak), seconds."""
    sig = movement_signal(w)
    sig = sig - sig.mean()
    if np.std(sig) < 1e-6:
        return 0.0
    ac = np.correlate(sig, sig, mode="full")[len(sig) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo = int(0.5 * VIDEO_FPS)
    if len(ac) <= lo + 1:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:]))
    return lag / VIDEO_FPS


def window_features(windows):
    """(N, T, 18, 2) → (N, F) hand-made features.

    - 9 joint angles (knees, hips, elbows, shoulders, trunk lean): mean, std, min, max, p10, p90
    - angular speed per angle: mean and p90 (rad/s), so rushing shows up
    - left/right asymmetry for knees, hips, elbows, shoulders: mean and max
    - every joint's x and y: mean, std, min, max (range of motion and posture)
    - every joint's mean speed
    - tempo: dominant period of the movement (s)
    - the six EC3D-style measures (knee width, feet width, depth, lean, knee over toe, sag)
    """
    w = np.asarray(windows, dtype=np.float32)
    n = len(w)
    if n == 0:
        return np.zeros((0, len(feature_names())), np.float32)
    ang = joint_angles(torch.from_numpy(w)).numpy()                       # (N, T, 9)
    speed = np.abs(np.diff(ang, axis=1)) * VIDEO_FPS                      # (N, T-1, 9)
    asym = np.stack([np.abs(ang[:, :, l] - ang[:, :, r]) for l, r in _LR_PAIRS], axis=2)
    xy = w.reshape(n, w.shape[1], 36)
    jspeed = np.linalg.norm(np.diff(w, axis=1), axis=3).mean(axis=1) * VIDEO_FPS   # (N, 18)
    parts = [
        ang.mean(1), ang.std(1), ang.min(1), ang.max(1),
        np.percentile(ang, 10, axis=1), np.percentile(ang, 90, axis=1),
        speed.mean(1), np.percentile(speed, 90, axis=1),
        asym.mean(1), asym.max(1),
        xy.mean(1), xy.std(1), xy.min(1), xy.max(1),
        jspeed,
        np.array([[_tempo_seconds(x)] for x in w]),
        np.stack([feature_vector(features_2d(x)) for x in w]),
    ]
    return np.nan_to_num(np.concatenate(parts, axis=1)).astype(np.float32)


def feature_names():
    angles = ["knee_r", "knee_l", "hip_r", "hip_l", "elbow_r", "elbow_l", "shoulder_r", "shoulder_l", "trunk"]
    names = []
    for stat in ("mean", "std", "min", "max", "p10", "p90"):
        names += [f"ang_{a}_{stat}" for a in angles]
    for stat in ("mean", "p90"):
        names += [f"angspeed_{a}_{stat}" for a in angles]
    for stat in ("mean", "max"):
        names += [f"asym_{p}_{stat}" for p in ("knee", "hip", "elbow", "shoulder")]
    for stat in ("mean", "std", "min", "max"):
        names += [f"j{j}_{c}_{stat}" for j in range(18) for c in "xy"]
    names += [f"j{j}_speed" for j in range(18)]
    names += ["tempo_s"]
    names += list(FEATURE_NAMES)
    return names


def proportion_features(windows):
    """Shortcut control: bone-length ratios only (camera placement / body shape, no movement)."""
    w = np.asarray(windows, dtype=np.float64)
    torso = np.linalg.norm(w[:, :, 1] - (w[:, :, 8] + w[:, :, 11]) / 2, axis=2) + 1e-6
    cols = [np.median(np.linalg.norm(w[:, :, a] - w[:, :, b], axis=2) / torso, axis=1) for a, b in _PROPORTION_BONES]
    return np.stack(cols, axis=1).astype(np.float32)


def angle_sequence(windows):
    """(N, T, 18, 2) → (N, 9 + 36, T) per-frame channels for the temporal CNN."""
    w = np.asarray(windows, dtype=np.float32)
    ang = joint_angles(torch.from_numpy(w)).numpy()                       # (N, T, 9)
    xy = w.reshape(len(w), w.shape[1], 36)
    return np.concatenate([ang, xy], axis=2).transpose(0, 2, 1).astype(np.float32)
