"""Parse BLE Heart Rate Measurement (UUID 0x2A37)."""
from __future__ import annotations


def parse_hr_measurement(payload: bytes | bytearray) -> tuple[float, list[float]]:
    """Return (bpm, rr_intervals_ms).

    Flags bit0: 0 = uint8 bpm, 1 = uint16 bpm (little-endian).
    Flags bit4: RR-Interval present; each RR is uint16 in units of 1/1024 s.
    """
    data = bytes(payload)
    if not data:
        raise ValueError("empty HR payload")
    flags = data[0]
    idx = 1
    if flags & 0x01:
        if len(data) < 3:
            raise ValueError("uint16 HR truncated")
        bpm = float(int.from_bytes(data[1:3], "little"))
        idx = 3
    else:
        if len(data) < 2:
            raise ValueError("uint8 HR truncated")
        bpm = float(data[1])
        idx = 2
    # Skip energy expended if present (bit 3)
    if flags & 0x08:
        idx += 2
    rr_ms: list[float] = []
    if flags & 0x10:
        while idx + 1 < len(data):
            ticks = int.from_bytes(data[idx:idx + 2], "little")
            rr_ms.append(ticks * 1000.0 / 1024.0)
            idx += 2
    if not 30 <= bpm <= 230:
        raise ValueError(f"bpm out of range: {bpm}")
    return bpm, rr_ms
