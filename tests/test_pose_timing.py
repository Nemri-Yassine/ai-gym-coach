"""Our recordings: time comes from frames.csv, not from the video's fps."""
import numpy as np
import pytest

from awc.features.pose_video import resample, resample_times


def _uneven_capture(seconds=6.0, mean_fps=20.0, seed=0):
    """Phone stream at ~20 fps with jitter; one joint moves with a 2 s period."""
    rng = np.random.default_rng(seed)
    gaps = rng.uniform(0.7, 1.3, size=int(seconds * mean_fps)) / mean_fps
    t = np.concatenate([[0.0], np.cumsum(gaps)])
    t = t[t <= seconds]
    seq = np.zeros((len(t), 18, 2))
    seq[:, 8, 1] = np.sin(2 * np.pi * t / 2.0)
    return seq, t * 1000.0


def test_resample_times_keeps_real_durations():
    seq, times_ms = _uneven_capture()
    out = resample_times(seq, times_ms, fps_out=30)
    duration = (times_ms[-1] - times_ms[0]) / 1000.0
    assert len(out) == int(np.floor(duration * 30)) + 1
    t_out = np.arange(len(out)) / 30
    assert np.allclose(out[:, 8, 1], np.sin(2 * np.pi * t_out / 2.0), atol=0.05)


def test_assuming_30_fps_would_distort_durations():
    """The bug this guards against: treating a 20 fps stream as 30 fps shrinks time by a third."""
    seq, times_ms = _uneven_capture()
    wrong = resample(seq, fps_in=30, fps_out=30)
    real_seconds = (times_ms[-1] - times_ms[0]) / 1000.0
    assert len(wrong) / 30 < 0.75 * real_seconds


def test_resample_times_rejects_bad_clocks():
    seq, times_ms = _uneven_capture()
    with pytest.raises(ValueError):
        resample_times(seq, times_ms[:-1])
    bad = times_ms.copy()
    bad[5] = bad[4]
    with pytest.raises(ValueError):
        resample_times(seq, bad)
