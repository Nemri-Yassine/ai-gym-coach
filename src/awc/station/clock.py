"""NTP-style phone↔laptop clock offset (median of the 5 lowest-RTT samples)."""
from __future__ import annotations

from collections import deque
from statistics import median


class ClockSync:
    def __init__(self, keep=64):
        self.samples = deque(maxlen=keep)  # (rtt_ms, offset_ms, t1)

    def add(self, t0_ms: float, phone_ts_ms: float, t1_ms: float):
        rtt = float(t1_ms - t0_ms)
        if rtt < 0:
            return
        offset = float(phone_ts_ms) - (float(t0_ms) + float(t1_ms)) / 2.0
        self.samples.append((rtt, offset, float(t1_ms)))

    def estimate(self):
        """Return (offset_ms, median_rtt_ms) or (None, None) if no samples."""
        if not self.samples:
            return None, None
        best = sorted(self.samples, key=lambda s: s[0])[:5]
        return float(median(s[1] for s in best)), float(median(s[0] for s in best))

    def phone_to_pc(self, phone_ts_ms: float) -> float:
        offset, _ = self.estimate()
        if offset is None:
            return float(phone_ts_ms)
        return float(phone_ts_ms) - offset
