"""Kafka message schemas for pose, heart rate and coach events."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

POSE_TOPIC = "pose.keypoints"
HR_TOPIC = "watch.hr"
IMU_TOPIC = "watch.imu"
COACH_TOPIC = "coach.events"
COACH_TYPES = frozenset({"rep", "form", "risk", "say"})
BPM_MIN, BPM_MAX = 30.0, 230.0
N_JOINTS = 18


@dataclass
class PoseKeypoints:
    session: str
    frame: int
    pc_ts_ms: int
    kp: list  # [[x, y, visibility] × 18], COCO-18, image coordinates

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "PoseKeypoints":
        return cls(**json.loads(text))


@dataclass
class WatchHr:
    session: str
    pc_ts_ms: int
    bpm: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "WatchHr":
        return cls(**json.loads(text))


@dataclass
class CoachEvent:
    session: str
    pc_ts_ms: int
    type: str
    payload: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "CoachEvent":
        return cls(**json.loads(text))


def validate(msg: Any) -> None:
    """Raise ValueError when a message is incomplete or out of range."""
    if isinstance(msg, PoseKeypoints):
        if not msg.session:
            raise ValueError("pose.keypoints: missing session")
        if not isinstance(msg.frame, int):
            raise ValueError("pose.keypoints: frame must be int")
        if not isinstance(msg.pc_ts_ms, int):
            raise ValueError("pose.keypoints: pc_ts_ms must be int")
        if not isinstance(msg.kp, list) or len(msg.kp) != N_JOINTS:
            raise ValueError(f"pose.keypoints: expected {N_JOINTS} joints")
        for joint in msg.kp:
            if not isinstance(joint, (list, tuple)) or len(joint) != 3:
                raise ValueError("pose.keypoints: each joint is [x, y, visibility]")
        return
    if isinstance(msg, WatchHr):
        if not msg.session:
            raise ValueError("watch.hr: missing session")
        if not isinstance(msg.pc_ts_ms, int):
            raise ValueError("watch.hr: pc_ts_ms must be int")
        bpm = float(msg.bpm)
        if not (BPM_MIN <= bpm <= BPM_MAX):
            raise ValueError(f"watch.hr: bpm {bpm} outside {BPM_MIN}-{BPM_MAX}")
        return
    if isinstance(msg, CoachEvent):
        if not msg.session:
            raise ValueError("coach.events: missing session")
        if not isinstance(msg.pc_ts_ms, int):
            raise ValueError("coach.events: pc_ts_ms must be int")
        if msg.type not in COACH_TYPES:
            raise ValueError(f"coach.events: type must be one of {sorted(COACH_TYPES)}")
        if not isinstance(msg.payload, dict):
            raise ValueError("coach.events: payload must be a dict")
        return
    raise ValueError(f"unknown message type {type(msg)!r}")
