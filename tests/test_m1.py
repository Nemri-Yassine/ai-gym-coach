"""Model 1 shapes, a few optimizer steps, and the --quick training entry point."""
import subprocess
import sys

import torch
from torch import nn

from awc.models.m1_recognition import CNN1DEncoder, LSTMEncoder, M1Model, TransformerEncoder


def test_encoders_shapes():
    x = torch.randn(4, 90, 36)
    assert CNN1DEncoder(36)(x).shape == (4, 128)
    assert LSTMEncoder(36)(x).shape == (4, 128)
    assert TransformerEncoder(36)(x).shape == (4, 64)


def test_m1_forward_all_inputs():
    pose = torch.randn(4, 90, 36)
    imu = torch.randn(4, 90, 6)
    for arch in ("cnn", "lstm", "transformer"):
        assert M1Model(arch, "pose")(pose).shape == (4, 11)
        assert M1Model(arch, "imu")(imu).shape == (4, 11)
        both = M1Model(arch, "both").eval()
        assert both(pose, imu, torch.zeros(4)).shape == (4, 11)
        both.train()
        assert both(pose, imu, torch.ones(4)).shape == (4, 11)


def test_one_step_decreases_loss():
    torch.manual_seed(0)
    model = M1Model("cnn", "pose")
    x = torch.randn(8, 90, 36)
    y = torch.randint(0, 11, (8,))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    def loss_value():
        model.eval()
        with torch.no_grad():
            return float(nn.functional.cross_entropy(model(x), y))

    before = loss_value()
    model.train()
    for _ in range(10):
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    assert loss_value() < before


def test_train_quick():
    subprocess.check_call([sys.executable, "-m", "awc.training.train_m1", "--arch", "cnn", "--inputs", "pose", "--quick"])
