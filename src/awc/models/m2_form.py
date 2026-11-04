"""Model 2: is this rep correct, or which mistake.

ST-GCN reads the skeleton graph. AngleLSTM reads nine joint angles.
Both have one head per exercise (squat 5, lunge 3, plank 3). The exercise
is known, so the loss uses only that head.
"""
import numpy as np
import torch
from torch import nn

from awc.features.form_dataset import EXERCISES, N_MISTAKES
from awc.features.skeleton import BONES

N_JOINTS = 18

# Angle at the middle joint: knees, hips, elbows, shoulders. Trunk lean is separate.
ANGLE_TRIPLES = (
    (8, 9, 10),
    (11, 12, 13),
    (2, 8, 9),
    (5, 11, 12),
    (2, 3, 4),
    (5, 6, 7),
    (1, 2, 3),
    (1, 5, 6),
)


def normalized_adjacency() -> np.ndarray:
    """BONES plus self-loops, then D^{-1/2} A D^{-1/2}. Shape (18, 18), symmetric."""
    a = np.eye(N_JOINTS, dtype=np.float64)
    for i, j in BONES:
        a[i, j] = 1.0
        a[j, i] = 1.0
    deg = a.sum(axis=1)
    scale = deg ** -0.5
    return (scale[:, None] * a * scale[None, :]).astype(np.float32)


def _angle_at(seq, a, b, c):
    """seq (..., 18, 2) → angle at joint b, in radians."""
    ba = seq[..., a, :] - seq[..., b, :]
    bc = seq[..., c, :] - seq[..., b, :]
    cos = (ba * bc).sum(dim=-1) / (ba.norm(dim=-1) * bc.norm(dim=-1) + 1e-8)
    return torch.acos(cos.clamp(-1.0, 1.0))


def joint_angles(seq) -> torch.Tensor:
    """seq (B, T, 18, 2), y up → (B, T, 9) angles in radians.

    Eight joint angles plus the trunk's lean away from vertical.
    """
    angles = [_angle_at(seq, a, b, c) for a, b, c in ANGLE_TRIPLES]
    mid = (seq[..., 8, :] + seq[..., 11, :]) / 2
    trunk = seq[..., 1, :] - mid
    vertical = torch.zeros_like(trunk)
    vertical[..., 1] = 1.0
    cos = (trunk * vertical).sum(dim=-1) / (trunk.norm(dim=-1) + 1e-8)
    angles.append(torch.acos(cos.clamp(-1.0, 1.0)))
    return torch.stack(angles, dim=-1)


class STGCNBlock(nn.Module):
    """Spatial graph conv (X·A·W), then a temporal conv of kernel 9."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.register_buffer("A", torch.from_numpy(normalized_adjacency()))
        self.spatial = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.bn_s = nn.BatchNorm2d(c_out)
        self.temporal = nn.Conv2d(
            c_out, c_out, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0),
        )
        self.bn_t = nn.BatchNorm2d(c_out)
        if c_in == c_out and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x):
        # x: (B, C, T, V)
        res = self.residual(x)
        y = torch.matmul(x, self.A)
        y = self.spatial(y)
        y = torch.relu(self.bn_s(y))
        y = self.bn_t(self.temporal(y))
        return torch.relu(y + res)


class _ExerciseHeads(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Linear(in_dim, N_MISTAKES[name]) for name in EXERCISES
        })

    def forward(self, h, exercise):
        if exercise.shape[0] != h.shape[0]:
            raise ValueError("one exercise id per sample")
        return {name: head(h) for name, head in self.heads.items()}


class STGCN(nn.Module):
    """(B, 2, 64, 18) → one logit vector per exercise head."""

    def __init__(self):
        super().__init__()
        self.block1 = STGCNBlock(2, 64, stride=1)
        self.block2 = STGCNBlock(64, 64, stride=1)
        self.block3 = STGCNBlock(64, 128, stride=2)
        self.heads = _ExerciseHeads(128)

    def forward(self, x, exercise):
        h = self.block3(self.block2(self.block1(x)))
        h = h.mean(dim=(2, 3))
        return self.heads(h, exercise)


class AngleLSTM(nn.Module):
    """Nine joint angles per frame → 2-layer LSTM, hidden 64 → exercise heads."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(9, 64, num_layers=2, batch_first=True)
        self.heads = _ExerciseHeads(64)

    def forward(self, x, exercise):
        seq = x.permute(0, 2, 3, 1)
        h, _ = self.lstm(joint_angles(seq))
        return self.heads(h[:, -1], exercise)


def build_model(name):
    if name == "stgcn":
        return STGCN()
    if name == "lstm":
        return AngleLSTM()
    raise ValueError(f"unknown model {name}")
