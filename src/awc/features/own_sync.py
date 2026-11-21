"""Join heart rate onto own-recording pose frames (laptop clock)."""
from __future__ import annotations

import numpy as np

HR_JOIN_MS = 2000  # ±2 s nearest reading, else NaN + mask 0


def join_hr_nearest(pc_ts_ms, hr_rows, max_ms=HR_JOIN_MS):
    """Nearest HR on the laptop clock. Outside ±max_ms → NaN with mask 0.

    hr_rows: list of (pc_ts_ms, bpm).
    Returns (hr, hr_mask) arrays aligned with pc_ts_ms.
    """
    pc_ts_ms = np.asarray(pc_ts_ms, dtype=np.float64)
    hr = np.full(len(pc_ts_ms), np.nan, dtype=np.float64)
    mask = np.zeros(len(pc_ts_ms), dtype=np.int32)
    if not hr_rows:
        return hr, mask
    times = np.asarray([t for t, _ in hr_rows], dtype=np.float64)
    bpms = np.asarray([b for _, b in hr_rows], dtype=np.float64)
    for i, t in enumerate(pc_ts_ms):
        j = int(np.argmin(np.abs(times - t)))
        if abs(times[j] - t) <= max_ms:
            hr[i] = bpms[j]
            mask[i] = 1
    return hr, mask
