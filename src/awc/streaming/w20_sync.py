"""Approximate w20 RGB video ↔ MM-Fit sensor-frame mapping.

Measured on w20 only. Fits 8 of 10 sampled sets within 3 frames; sit-up and
tricep samples did not fit. Drift is ~0.10 s near the start and up to ~5.3 s
near the end. Use for streaming replay timing only — never for fatigue or
model evaluation scores.
"""
from awc.config import (
    W20_VIDEO_SENSOR_OFFSET,
    W20_VIDEO_SENSOR_PIVOT,
    W20_VIDEO_SENSOR_SLOPE,
)


def sensor_frame_from_video(video_frame: float) -> float:
    """sensor_frame = video_frame + 3 + 0.00169 * (video_frame - 2178). Approximate."""
    v = float(video_frame)
    return v + W20_VIDEO_SENSOR_OFFSET + W20_VIDEO_SENSOR_SLOPE * (v - W20_VIDEO_SENSOR_PIVOT)


def video_frame_from_sensor(sensor_frame: float) -> float:
    """Inverse of sensor_frame_from_video. Approximate."""
    s = float(sensor_frame)
    return (
        s - W20_VIDEO_SENSOR_OFFSET + W20_VIDEO_SENSOR_SLOPE * W20_VIDEO_SENSOR_PIVOT
    ) / (1.0 + W20_VIDEO_SENSOR_SLOPE)
