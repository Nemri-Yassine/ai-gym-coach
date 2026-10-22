"""Model 1: which exercise, from a 3-second window of pose and/or watch IMU.

Three encoders share the interface forward(x: (B, T, C)) -> (B, D).
With inputs='both', 30% of training samples drop the IMU so the same checkpoint
can run on pose only.
"""
import torch
from torch import nn

N_CLASSES = 11
POSE_DIM = 36
IMU_DIM = 6
MODALITY_DROPOUT = 0.3


class CNN1DEncoder(nn.Module):
    """3 conv blocks, channels 64-128-128, global average pool → 128."""

    def __init__(self, in_ch):
        super().__init__()
        channels = (in_ch, 64, 128, 128)
        blocks = []
        for a, b in zip(channels[:-1], channels[1:]):
            blocks += [
                nn.Conv1d(a, b, kernel_size=5, padding=2),
                nn.BatchNorm1d(b),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        h = self.net(x.transpose(1, 2))
        return h.mean(dim=-1)


class LSTMEncoder(nn.Module):
    """2-layer bidirectional LSTM, hidden 64, mean over time → 128."""

    def __init__(self, in_ch):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, 64, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x):
        h, _ = self.lstm(x)
        return h.mean(dim=1)


class TransformerEncoder(nn.Module):
    """Linear to 64, learned positions, 2 layers, 4 heads, mean pool → 64."""

    def __init__(self, in_ch, length=90):
        super().__init__()
        self.proj = nn.Linear(in_ch, 64)
        self.pos = nn.Parameter(torch.zeros(1, length, 64))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, x):
        h = self.proj(x) + self.pos[:, :x.shape[1]]
        return self.enc(h).mean(dim=1)


def make_encoder(arch, in_ch):
    if arch == "cnn":
        return CNN1DEncoder(in_ch)
    if arch == "lstm":
        return LSTMEncoder(in_ch)
    if arch == "transformer":
        return TransformerEncoder(in_ch)
    raise ValueError(f"unknown arch {arch}")


def encoder_dim(arch):
    return 64 if arch == "transformer" else 128


class M1Model(nn.Module):
    """inputs is 'pose', 'imu' or 'both'."""

    def __init__(self, arch, inputs):
        super().__init__()
        if inputs not in {"pose", "imu", "both"}:
            raise ValueError(inputs)
        self.arch = arch
        self.inputs = inputs
        if inputs in {"pose", "both"}:
            self.pose_enc = make_encoder(arch, POSE_DIM)
        if inputs in {"imu", "both"}:
            self.imu_enc = make_encoder(arch, IMU_DIM)
        d = encoder_dim(arch)
        if inputs == "both":
            in_dim = d + d + 1
        else:
            in_dim = d
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, x, x_imu=None, imu_mask=None):
        if self.inputs == "pose":
            return self.head(self.pose_enc(x))
        if self.inputs == "imu":
            return self.head(self.imu_enc(x))
        pose = self.pose_enc(x)
        imu = self.imu_enc(x_imu)
        if imu_mask is None:
            imu_mask = torch.ones(pose.shape[0], device=pose.device, dtype=pose.dtype)
        else:
            imu_mask = imu_mask.to(device=pose.device, dtype=pose.dtype).reshape(-1)
        if self.training:
            keep = torch.rand(pose.shape[0], device=pose.device) >= MODALITY_DROPOUT
            imu_mask = imu_mask * keep.to(pose.dtype)
        fused = torch.cat([pose, imu * imu_mask[:, None], imu_mask[:, None]], dim=1)
        return self.head(fused)
