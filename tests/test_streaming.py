"""Unit + Kafka integration tests for streaming schemas and producers."""
from __future__ import annotations

import json
import statistics
import time

import cv2
import pytest

from awc.config import POSE_MODEL, ROOT
from awc.streaming.kafka_io import ensure_topics, make_producer, publish
from awc.streaming.schemas import (
    CoachEvent,
    PoseKeypoints,
    WatchHr,
    validate,
)

ASSETS = ROOT / "tests" / "assets"
STANDING_IMAGE = ASSETS / "standing_person.jpg"


def _pose(n_joints=18, **overrides):
    kp = [[0.0, 1.0, 0.9] for _ in range(n_joints)]
    base = dict(session="s", frame=0, pc_ts_ms=1, kp=kp)
    base.update(overrides)
    return PoseKeypoints(**base)


def test_schema_roundtrip():
    pose = _pose()
    assert PoseKeypoints.from_json(pose.to_json()) == pose
    hr = WatchHr(session="s", pc_ts_ms=10, bpm=72.5)
    assert WatchHr.from_json(hr.to_json()) == hr
    ev = CoachEvent(session="s", pc_ts_ms=20, type="say", payload={"text": "Stop"})
    assert CoachEvent.from_json(ev.to_json()) == ev


def test_validate_rejects_bad_joint_counts():
    with pytest.raises(ValueError, match="18"):
        validate(_pose(n_joints=17))


def test_validate_rejects_missing_fields():
    with pytest.raises(ValueError, match="session"):
        validate(PoseKeypoints(session="", frame=0, pc_ts_ms=1, kp=[[0, 0, 1]] * 18))
    with pytest.raises(ValueError, match="session"):
        validate(WatchHr(session="", pc_ts_ms=1, bpm=70))
    with pytest.raises(ValueError, match="type"):
        validate(CoachEvent(session="s", pc_ts_ms=1, type="nope", payload={}))


def test_validate_rejects_bpm_out_of_range():
    with pytest.raises(ValueError, match="bpm"):
        validate(WatchHr(session="s", pc_ts_ms=1, bpm=20))
    with pytest.raises(ValueError, match="bpm"):
        validate(WatchHr(session="s", pc_ts_ms=1, bpm=240))
    validate(WatchHr(session="s", pc_ts_ms=1, bpm=60))


@pytest.mark.skipif(not STANDING_IMAGE.is_file(), reason="tests/assets/standing_person.jpg missing")
@pytest.mark.skipif(not POSE_MODEL.is_file(), reason="pose landmarker model missing")
def test_mediapipe_mapping_on_image():
    from awc.streaming.pose_producer import _open_landmarker, frame_to_message

    bgr = cv2.imread(str(STANDING_IMAGE))
    assert bgr is not None
    with _open_landmarker() as landmarker:
        msg = frame_to_message("asset", 0, 0, bgr, landmarker)
    assert msg is not None
    assert len(msg.kp) == 18
    # nose (0) and neck (1) should sit above ankles (10, 13) when y points up
    head_y = max(msg.kp[0][1], msg.kp[1][1])
    ankle_y = min(msg.kp[10][1], msg.kp[13][1])
    assert head_y > ankle_y


@pytest.mark.kafka
def test_kafka_pose_roundtrip_latency():
    import threading
    from kafka import KafkaConsumer, TopicPartition

    from awc.config import KAFKA_BOOTSTRAP

    ensure_topics()
    producer = make_producer()
    # Manual assign avoids group-coordinator races on a fresh Docker broker.
    consumer = KafkaConsumer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        enable_auto_commit=False,
        consumer_timeout_ms=1000,
        value_deserializer=lambda b: b.decode("utf-8"),
    )
    tp = TopicPartition("pose.keypoints", 0)
    consumer.assign([tp])
    consumer.seek_to_end(tp)
    consumer.position(tp)

    n = 100
    session = f"kafka_test_{int(time.time())}"
    sent_frames = list(range(n))
    got = []
    latencies = []
    stop = threading.Event()

    def _consume():
        while not stop.is_set() and len(got) < n:
            polled = consumer.poll(timeout_ms=200)
            now = time.time() * 1000.0
            for records in polled.values():
                for record in records:
                    payload = json.loads(record.value)
                    if payload.get("session") != session:
                        continue
                    got.append(payload["frame"])
                    latencies.append(now - float(payload["pc_ts_ms"]))

    thread = threading.Thread(target=_consume, daemon=True)
    thread.start()
    time.sleep(0.2)
    for i in sent_frames:
        msg = _pose(frame=i, pc_ts_ms=int(time.time() * 1000), session=session)
        publish(producer, "pose.keypoints", msg)
    producer.flush()
    deadline = time.time() + 20.0
    while len(got) < n and time.time() < deadline:
        time.sleep(0.05)
    stop.set()
    thread.join(timeout=2)
    producer.close()
    consumer.close()
    assert got == sent_frames
    assert latencies and statistics.median(latencies) < 200.0, (
        None if not latencies else statistics.median(latencies)
    )
