"""Model 3: small GRU over reps → fatigue (MSE) and risk (BCE) heads."""
from __future__ import annotations

import torch
from torch import nn


class M3Model(nn.Module):
    def __init__(self, in_dim=10, hidden=32):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=1, batch_first=True)
        self.fatigue = nn.Linear(hidden, 1)
        self.risk = nn.Linear(hidden, 1)

    def forward(self, x, lengths):
        """x: (B, T, C), lengths: (B,) int. Returns fatigue, risk sigmoid (B, T)."""
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        fat = torch.sigmoid(self.fatigue(out)).squeeze(-1)
        risk = torch.sigmoid(self.risk(out)).squeeze(-1)
        return fat, risk


def rule_baseline_alert(speeds, drop=0.25):
    """B1: first index where speed loss > drop vs mean of first 2 reps. None if never."""
    speeds = list(speeds)
    if len(speeds) < 3:
        return None
    base = sum(speeds[:2]) / 2.0
    if base <= 0:
        return None
    for i in range(2, len(speeds)):
        if speeds[i] <= (1.0 - drop) * base:
            return i
    return None


def pad_batch(xs, masks=None):
    """Pad list of (T, C) to (B, Tmax, C) + lengths."""
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    tmax = int(lengths.max())
    c = xs[0].shape[-1]
    out = torch.zeros(len(xs), tmax, c, dtype=torch.float32)
    for i, x in enumerate(xs):
        out[i, : len(x)] = torch.as_tensor(x, dtype=torch.float32)
    mask_out = None
    if masks is not None:
        mc = masks[0].shape[-1]
        mask_out = torch.zeros(len(xs), tmax, mc, dtype=torch.float32)
        for i, m in enumerate(masks):
            mask_out[i, : len(m)] = torch.as_tensor(m, dtype=torch.float32)
    return out, lengths, mask_out
