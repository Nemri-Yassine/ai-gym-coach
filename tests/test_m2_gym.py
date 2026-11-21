"""Model 2 v2 helpers: windows, features, relative verdict (no trained model needed)."""
import numpy as np

from awc.features.form_windows import HOP, WINDOW, feature_names, set_windows, window_features
from awc.models.m2_gym import RELATIVE_DROP, relative_verdict


def _fake_set(seconds=12, fps=30, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * fps) / fps
    base = rng.normal(0, 1, size=(18, 2)).astype(np.float32)
    move = np.sin(2 * np.pi * t / 2.0)[:, None, None] * 0.3
    return (base[None] + move + rng.normal(0, 0.01, size=(len(t), 18, 2))).astype(np.float32)


def test_windows_shape_and_hop():
    w = set_windows(_fake_set(), trim=False)
    assert w.shape[1:] == (WINDOW, 18, 2)
    assert len(w) == (12 * 30 - WINDOW) // HOP + 1


def test_windows_drop_nan():
    pose = _fake_set()
    pose[100] = np.nan
    w = set_windows(pose, trim=False)
    assert np.isfinite(w).all()


def test_feature_count_matches_names():
    f = window_features(set_windows(_fake_set(), trim=False)[:3])
    assert f.shape == (3, len(feature_names()))
    assert np.isfinite(f).all()


def test_relative_verdict():
    assert relative_verdict(0.5, []) is None
    assert relative_verdict(0.80, [0.82, 0.78, 0.85])["verdict"] == "as usual"
    worse = relative_verdict(0.80 - RELATIVE_DROP - 0.05, [0.80, 0.81, 0.79])
    assert worse["verdict"] == "worse than usual" and worse["n_earlier"] == 3
