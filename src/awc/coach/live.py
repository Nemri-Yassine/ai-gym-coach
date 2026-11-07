"""Replay coach: offline Parquet or Kafka consumers → M1→reps→M3→policy.

    python -m awc.coach.live --replay w20 --no-voice
    python -m awc.coach.live --kafka --replay w20 --no-voice --max-seconds 90
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import deque

import numpy as np
import torch

from awc.config import KAFKA_BOOTSTRAP, MMFIT_CLASSES, MODELS, VIDEO_FPS
from awc.coach.policy import Policy
from awc.coach.voice import Voice
from awc.features.reps import count_reps
from awc.features.skeleton import normalize
from awc.models.m1_recognition import M1Model
from awc.models.m3_fatigue import M3Model
from awc.streaming.kafka_io import ensure_topics, make_consumer, make_producer, publish
from awc.streaming.schemas import COACH_TOPIC, HR_TOPIC, POSE_TOPIC, CoachEvent


def _load_m1():
    path = MODELS / "m1_transformer_pose.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = M1Model(ckpt["arch"], ckpt["inputs"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _load_m3():
    path = MODELS / "m3_b4.pt"
    if not path.is_file():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = M3Model(in_dim=ckpt.get("in_dim", 10), hidden=32)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, float(ckpt.get("thresh", 0.7))


def _decide_on_set(pose_n, exercise, m3, policy, voice):
    """Run rep loop on one normalized set. Returns messages + model latencies (ms)."""
    count, bounds = count_reps(pose_n, fps=VIDEO_FPS, exercise=exercise)
    set_feats = []
    messages = []
    model_latencies = []
    for ri, (a, b) in enumerate(bounds):
        t0 = time.perf_counter()
        duration = (b - a) / VIDEO_FPS
        rom = float(np.ptp(pose_n[a:b + 1, 8, 1]))
        speed = rom / max(duration, 1e-3)
        set_feats.append([duration, rom, speed, 0.5, 0.0, 0.0, ri / max(count - 1, 1), ri * duration])
        risk = 0.0
        if m3 is not None and len(set_feats) >= 2:
            x = torch.tensor([set_feats], dtype=torch.float32)
            if x.shape[-1] < 10:
                x = torch.cat([x, torch.zeros(1, x.shape[1], 10 - x.shape[-1])], dim=-1)
            with torch.no_grad():
                _f, r = m3(x, torch.tensor([len(set_feats)]))
                risk = float(r[0, -1])
        decision = policy.decide(risk=risk)
        dt_ms = (time.perf_counter() - t0) * 1000
        model_latencies.append(dt_ms)
        if decision.type != "none":
            messages.append({
                "rep": ri, "msg": decision.message, "risk": risk,
                "model_latency_ms": dt_ms,
            })
            voice.say(decision.message)
            print(f"rep{ri}: {decision.message} (risk={risk:.2f}, model {dt_ms:.0f} ms)", flush=True)
    return messages, model_latencies


def run_replay_offline(session="w20", no_voice=True, max_sets=3):
    """Offline demo from interim synced pose (no Kafka). Model-only latency."""
    import pandas as pd
    from awc.config import INTERIM
    from awc.data import mmfit
    from awc.features.sync import POSE_X, POSE_Y

    path = INTERIM / "mmfit_synced" / f"session={session}"
    df = pd.read_parquet(path)
    if "session" not in df.columns:
        df["session"] = session
    m1 = _load_m1()
    m3_pack = _load_m3()
    m3, alert_thresh = (None, 0.7) if m3_pack is None else m3_pack
    # slow band sits below the val-tuned stop threshold (hardcoded 0.4 was from the old default)
    slow_thresh = min(0.4, max(0.25, alert_thresh - 0.15))
    policy = Policy(stop_thresh=alert_thresh, slow_thresh=slow_thresh)
    voice = Voice(enabled=not no_voice)
    all_msgs, all_lat = [], []
    labels = mmfit.load_labels(session)[:max_sets]
    for set_id, (s0, s1, n_reps, exercise) in enumerate(labels):
        block = df[(df["frame"] >= s0) & (df["frame"] <= s1)].sort_values("frame")
        if len(block) < 30:
            continue
        pose = np.stack([block[POSE_X].to_numpy(), block[POSE_Y].to_numpy()], axis=-1)
        pose_n = normalize(pose)
        msgs, lats = _decide_on_set(pose_n, exercise, m3, policy, voice)
        for m in msgs:
            m["set"] = set_id
        all_msgs.extend(msgs)
        all_lat.extend(lats)
    voice.close()
    return {
        "session": session,
        "mode": "offline_parquet",
        "messages": all_msgs,
        "median_model_latency_ms": statistics.median(all_lat) if all_lat else None,
        "note": "model-only latency; excludes pose extraction and Kafka",
        "n_reps": len(all_lat),
        "alert_thresh": alert_thresh,
    }


def run_kafka(session="w20", no_voice=True, max_seconds=90, bootstrap=KAFKA_BOOTSTRAP):
    """Consume pose.keypoints (+ optional HR), buffer, detect set-like motion, coach, publish coach.events.

    End-to-end latency: frame pc_ts_ms → message time.
    Model latency: rep end → message (inference only).
    """
    ensure_topics(bootstrap)
    m1 = _load_m1()
    m3_pack = _load_m3()
    m3, alert_thresh = (None, 0.7) if m3_pack is None else m3_pack
    slow_thresh = min(0.4, max(0.25, alert_thresh - 0.15))
    policy = Policy(stop_thresh=alert_thresh, slow_thresh=slow_thresh)
    voice = Voice(enabled=not no_voice)
    producer = make_producer(bootstrap)
    consumer = make_consumer(f"awc-coach-{int(time.time())}", [POSE_TOPIC, HR_TOPIC], bootstrap=bootstrap)

    ring = deque(maxlen=90 * 30)  # up to ~90 s of keypoints at 30 Hz
    frame_ts = deque(maxlen=90 * 30)
    messages = []
    model_lats, e2e_lats = [], []
    last_exercise = "non_activity"
    t0 = time.perf_counter()
    last_acted_end = -1
    n_pose_msgs = 0
    try:
        while time.perf_counter() - t0 < max_seconds:
            polled = consumer.poll(timeout_ms=200)
            new_frames = 0
            for _tp, records in polled.items():
                for record in records:
                    payload = json.loads(record.value)
                    if payload.get("session") not in (None, session):
                        continue
                    if record.topic != POSE_TOPIC:
                        continue
                    kp = payload.get("kp")
                    if not kp or len(kp) != 18:
                        continue
                    xy = np.array([[p[0], p[1]] for p in kp], dtype=np.float64)
                    ring.append(xy)
                    frame_ts.append(float(payload.get("pc_ts_ms", time.time() * 1000)))
                    new_frames += 1
                    n_pose_msgs += 1

            if new_frames == 0 or len(ring) < 90:
                continue
            # Re-evaluate roughly once per second of new frames
            if n_pose_msgs % 30 != 0:
                continue
            pose = np.stack(list(ring), axis=0)
            finite = np.isfinite(pose).all(axis=(1, 2))
            if finite.mean() < 0.7:
                continue
            pose = np.nan_to_num(pose, nan=0.0)
            pose_n = normalize(pose)
            win = pose_n[-90:]
            with torch.no_grad():
                logits = m1(torch.from_numpy(win.reshape(1, 90, 36).astype(np.float32)))
                last_exercise = MMFIT_CLASSES[int(logits.argmax())]
            if last_exercise in ("non_activity",):
                continue
            count, bounds = count_reps(pose_n, fps=VIDEO_FPS, exercise=last_exercise)
            if count < 2 or not bounds:
                continue
            a, b = bounds[-1]
            if b <= last_acted_end:
                continue
            # Require the latest rep to have finished recently in the buffer
            if b < len(pose_n) - 45:
                continue
            last_acted_end = b
            t_model0 = time.perf_counter()
            set_feats = []
            for ri, (ba, bb) in enumerate(bounds):
                duration = (bb - ba) / VIDEO_FPS
                rom = float(np.ptp(pose_n[ba:bb + 1, 8, 1]))
                speed = rom / max(duration, 1e-3)
                set_feats.append([duration, rom, speed, 0.5, 0.0, 0.0, ri / max(count - 1, 1), ri * duration])
            risk = 0.0
            if m3 is not None and len(set_feats) >= 2:
                x = torch.tensor([set_feats], dtype=torch.float32)
                if x.shape[-1] < 10:
                    x = torch.cat([x, torch.zeros(1, x.shape[1], 10 - x.shape[-1])], dim=-1)
                with torch.no_grad():
                    _f, r = m3(x, torch.tensor([len(set_feats)]))
                    risk = float(r[0, -1])
            decision = policy.decide(risk=risk)
            model_ms = (time.perf_counter() - t_model0) * 1000
            now_ms = time.time() * 1000
            frame_idx = min(b, len(frame_ts) - 1)
            e2e_ms = now_ms - frame_ts[frame_idx]
            model_lats.append(model_ms)
            e2e_lats.append(e2e_ms)
            msg = {
                "rep": count - 1, "msg": decision.message or "(none)", "risk": risk,
                "model_latency_ms": model_ms, "e2e_latency_ms": e2e_ms,
                "exercise": last_exercise, "type": decision.type,
            }
            if decision.type != "none":
                messages.append(msg)
                voice.say(decision.message)
                print(
                    f"{last_exercise} rep{count}: {decision.message} "
                    f"(risk={risk:.2f}, model {model_ms:.0f} ms, e2e {e2e_ms:.0f} ms)",
                    flush=True,
                )
                event = CoachEvent(
                    session=session, pc_ts_ms=int(now_ms), type="say",
                    payload={"text": decision.message, "risk": risk, "exercise": last_exercise},
                )
                try:
                    publish(producer, COACH_TOPIC, event)
                    producer.flush()
                except Exception as exc:  # noqa: BLE001
                    print(f"publish failed: {exc}", flush=True)
            else:
                print(
                    f"{last_exercise} rep{count}: (none) "
                    f"(risk={risk:.2f}, model {model_ms:.0f} ms, e2e {e2e_ms:.0f} ms)",
                    flush=True,
                )
    finally:
        consumer.close()
        producer.close()
        voice.close()
    return {
        "session": session,
        "mode": "kafka",
        "messages": messages,
        "median_model_latency_ms": statistics.median(model_lats) if model_lats else None,
        "median_e2e_latency_ms": statistics.median(e2e_lats) if e2e_lats else None,
        "n_reps": len(model_lats),
        "n_pose_msgs": n_pose_msgs,
        "alert_thresh": alert_thresh,
        "slow_thresh": slow_thresh,
        "seconds": max_seconds,
        "model_latencies_ms": model_lats,
        "e2e_latencies_ms": e2e_lats,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--replay", default="w20")
    p.add_argument("--no-voice", action="store_true")
    p.add_argument("--max-sets", type=int, default=3)
    p.add_argument("--max-seconds", type=float, default=90)
    p.add_argument("--kafka", action="store_true", help="Consume Kafka pose/HR from 2.3 producers")
    p.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP)
    args = p.parse_args(argv)
    if args.kafka:
        stats = run_kafka(args.replay, no_voice=args.no_voice, max_seconds=args.max_seconds, bootstrap=args.bootstrap)
    else:
        stats = run_replay_offline(args.replay, no_voice=args.no_voice, max_sets=args.max_sets)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
