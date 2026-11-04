"""Model 2: graph, angles, and the quick training entry points."""
import subprocess
import sys

import numpy as np
import torch
from torch import nn

from awc.features.form_features import FEATURE_NAMES, feature_vector, features_2d
from awc.features.skeleton import BONES
from awc.models.m2_form import STGCN, normalized_adjacency


def test_adjacency_symmetric_normalized():
    a = normalized_adjacency()
    assert a.shape == (18, 18)
    assert a.dtype == np.float32
    np.testing.assert_allclose(a, a.T, atol=1e-6)
    raw = np.eye(18, dtype=np.float64)
    for i, j in BONES:
        raw[i, j] = 1.0
        raw[j, i] = 1.0
    scale = raw.sum(axis=1) ** -0.5
    expected = scale[:, None] * raw * scale[None, :]
    np.testing.assert_allclose(a, expected, atol=1e-6)
    assert np.isfinite(a).all()


def test_stgcn_forward():
    model = STGCN()
    x = torch.randn(8, 2, 64, 18)
    exercise = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2])
    out = model(x, exercise)
    assert out["squat"].shape == (8, 5)
    assert out["lunge"].shape == (8, 3)
    assert out["plank"].shape == (8, 3)
    assert torch.isfinite(out["squat"]).all()


def test_overfit_tiny():
    torch.manual_seed(0)
    model = STGCN()
    x = torch.zeros(16, 2, 64, 18)
    y = torch.arange(16) % 5
    exercise = torch.zeros(16, dtype=torch.long)
    for i in range(16):
        x[i, :, :, int(y[i])] = 1.0
        x[i, :, int(y[i]), :] = float(y[i] + 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(100):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x, exercise)["squat"], y)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(x, exercise)["squat"].argmax(dim=1)
    assert torch.equal(pred, y)


def test_features_2d_finite():
    rng = np.random.default_rng(0)
    seq = rng.normal(size=(20, 18, 2))
    seq[:, 1, 1] += 2.0
    measures = features_2d(seq)
    assert set(measures) == set(FEATURE_NAMES)
    vector = feature_vector(measures)
    assert vector.shape == (6,)
    assert np.isfinite(vector).all()


def test_train_quick():
    for model in ("stgcn", "rf"):
        subprocess.check_call(
            [sys.executable, "-m", "awc.training.train_m2", "--model", model, "--quick"],
        )
