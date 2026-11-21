"""Unit tests for video↔sensor sync fitting (no dataset required)."""
import numpy as np

from awc.features.video_sync import (
    cross_corr_offset,
    fit_linear_sync,
    fit_session_from_set_offsets,
    sensor_from_video,
    video_from_sensor,
)


def test_cross_corr_known_offset():
    # query is ref delayed by 25 samples: query[t] ≈ ref[t - 25] for t >= 25
    # → best lag so query[t] ≈ ref[t + lag] is lag = -25
    t = np.linspace(0, 8 * np.pi, 400)
    ref = np.sin(t)
    delay = 25
    query = np.zeros_like(ref)
    query[delay:] = ref[:-delay]
    est = cross_corr_offset(ref, query, max_lag=80)
    assert abs(est + delay) <= 1


def test_linear_fit_known_offset_and_drift():
    rng = np.random.default_rng(42)
    video = np.linspace(1000, 5000, 12)
    a_true, b_true, pivot = 3.0, 0.002, 2000.0
    sensor = video + a_true + b_true * (video - pivot) + rng.normal(0, 0.05, size=len(video))
    a, b, p = fit_linear_sync(video, sensor, pivot=pivot)
    assert abs(a - a_true) < 0.05
    assert abs(b - b_true) < 5e-4
    for v, s in zip(video, sensor):
        assert abs(sensor_from_video(v, a, b, p) - s) < 0.3
        assert abs(video_from_sensor(s, a, b, p) - v) < 0.3


def test_fit_session_within_3():
    video = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
    a, b, pivot = 2.0, 0.001, 2500.0
    sensor = np.array([sensor_from_video(v, a, b, pivot) for v in video])
    fit = fit_session_from_set_offsets(video, sensor, pivot=pivot)
    assert fit["n_within_3"] == 5
    assert fit["frac_within_3"] == 1.0
