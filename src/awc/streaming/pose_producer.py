"""Publish MediaPipe COCO-18 keypoints to pose.keypoints.

    python -m awc.streaming.pose_producer --video path.mp4 --session w20_replay
    python -m awc.streaming.pose_producer --camera-url http://PHONE:8080/video --session live1

With --video the file is replayed at the video's fps (demo / streaming safety net).
Coordinates are pixels with y up (project convention after MediaPipe mapping).
"""
from __future__ import annotations

import argparse
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from awc.config import KAFKA_BOOTSTRAP, POSE_MODEL, REPLAY_POSE_VIDEO, VIDEO_FPS
from awc.data.mmfit import JOINTS_2D
from awc.features.pose_video import landmarks_to_pixels
from awc.features.skeleton import MEDIAPIPE_TO_COCO18
from awc.streaming.kafka_io import ensure_topics, make_producer, publish
from awc.streaming.schemas import POSE_TOPIC, PoseKeypoints


def _visibility_coco18(landmarks) -> list[float]:
    """Map MediaPipe visibility onto COCO-18. Neck = mean of the two shoulders."""
    out = []
    for name in JOINTS_2D:
        spec = MEDIAPIPE_TO_COCO18[name]
        if name == "neck":
            a, b = spec
            out.append(0.5 * (float(landmarks[a].visibility) + float(landmarks[b].visibility)))
        else:
            out.append(float(landmarks[spec].visibility))
    return out


def frame_to_message(session, frame_idx, pc_ts_ms, bgr, landmarker) -> PoseKeypoints | None:
    height, width = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(image, int(pc_ts_ms))
    if not result.pose_landmarks:
        kp = [[float("nan"), float("nan"), 0.0] for _ in range(18)]
        return PoseKeypoints(session=session, frame=int(frame_idx), pc_ts_ms=int(pc_ts_ms), kp=kp)
    lm = result.pose_landmarks[0]
    xy = landmarks_to_pixels([(p.x, p.y) for p in lm], width, height)
    vis = _visibility_coco18(lm)
    kp = [[float(xy[i, 0]), float(xy[i, 1]), float(vis[i])] for i in range(18)]
    return PoseKeypoints(session=session, frame=int(frame_idx), pc_ts_ms=int(pc_ts_ms), kp=kp)


def _open_landmarker(model_path=POSE_MODEL):
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def run_video(path, session, bootstrap=KAFKA_BOOTSTRAP, max_seconds=None, realtime=True):
    """Replay a video file. Returns stats dict. Does not score models."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS)
    if not np.isfinite(fps) or fps <= 1:
        fps = float(VIDEO_FPS)
    frame_period = 1.0 / fps
    ensure_topics(bootstrap)
    producer = make_producer(bootstrap)
    sent = 0
    t0 = time.perf_counter()
    wall0 = time.time() * 1000.0
    try:
        with _open_landmarker() as landmarker:
            index = 0
            next_due = time.perf_counter()
            while True:
                if max_seconds is not None and (time.perf_counter() - t0) >= max_seconds:
                    break
                ok, bgr = cap.read()
                if not ok:
                    break
                if realtime:
                    now = time.perf_counter()
                    sleep_for = next_due - now
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    next_due += frame_period
                pc_ts_ms = int(wall0 + index * 1000.0 / fps)
                # Prefer wall clock at send so monitor latency reflects real pipeline delay.
                pc_ts_ms = int(time.time() * 1000)
                msg = frame_to_message(session, index, pc_ts_ms, bgr, landmarker)
                publish(producer, POSE_TOPIC, msg)
                sent += 1
                index += 1
            producer.flush()
    finally:
        cap.release()
        producer.close()
    elapsed = time.perf_counter() - t0
    return {
        "session": session,
        "path": str(path),
        "frames": sent,
        "fps_source": fps,
        "seconds": elapsed,
        "msg_per_s": sent / elapsed if elapsed > 0 else 0.0,
    }


def run_camera(url, session, bootstrap=KAFKA_BOOTSTRAP, max_seconds=None):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {url}")
    ensure_topics(bootstrap)
    producer = make_producer(bootstrap)
    sent = 0
    t0 = time.perf_counter()
    try:
        with _open_landmarker() as landmarker:
            index = 0
            while True:
                if max_seconds is not None and (time.perf_counter() - t0) >= max_seconds:
                    break
                ok, bgr = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                pc_ts_ms = int(time.time() * 1000)
                msg = frame_to_message(session, index, pc_ts_ms, bgr, landmarker)
                publish(producer, POSE_TOPIC, msg)
                sent += 1
                index += 1
            producer.flush()
    finally:
        cap.release()
        producer.close()
    elapsed = time.perf_counter() - t0
    return {
        "session": session,
        "path": url,
        "frames": sent,
        "seconds": elapsed,
        "msg_per_s": sent / elapsed if elapsed > 0 else 0.0,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Pose keypoints → Kafka")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Replay a video file at real speed")
    src.add_argument("--camera-url", help="Live OpenCV camera / IP Webcam URL")
    p.add_argument("--session", required=True)
    p.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--no-realtime", action="store_true", help="Do not sleep between frames")
    args = p.parse_args(argv)
    if args.video:
        path = REPLAY_POSE_VIDEO if args.video in ("w20", "default") else args.video
        stats = run_video(path, args.session, args.bootstrap, args.max_seconds, realtime=not args.no_realtime)
    else:
        stats = run_camera(args.camera_url, args.session, args.bootstrap, args.max_seconds)
    print(stats, flush=True)


if __name__ == "__main__":
    main()
