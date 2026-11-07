"""Replay coach: consume Kafka pose/HR (or offline parquet), run M1→reps→M2→M3→policy.

    python -m awc.coach.live --replay w20 --no-voice --max-seconds 60
"""
from __future__ import annotations

import argparse
import json
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
from awc.streaming.schemas import COACH_TOPIC, CoachEvent


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
    return model


def run_replay_offline(session="w20", no_voice=True, max_sets=3):
    """Offline demo from interim synced pose (no Kafka). Returns latency stats."""
    import pandas as pd
    from awc.config import INTERIM
    from awc.data import mmfit
    from awc.features.sync import POSE_X, POSE_Y

    path = INTERIM / "mmfit_synced" / f"session={session}"
    df = pd.read_parquet(path)
    if "session" not in df.columns:
        df["session"] = session
    m1 = _load_m1()
    m3 = _load_m3()
    policy = Policy()
    voice = Voice(enabled=not no_voice)
    latencies = []
    labels = mmfit.load_labels(session)[:max_sets]
    messages = []
    for set_id, (s0, s1, n_reps, exercise) in enumerate(labels):
        block = df[(df["frame"] >= s0) & (df["frame"] <= s1)].sort_values("frame")
        if len(block) < 30:
            continue
        pose = np.stack([block[POSE_X].to_numpy(), block[POSE_Y].to_numpy()], axis=-1)
        pose_n = normalize(pose)
        # M1 mid window
        mid = len(pose_n) // 2
        win = pose_n[max(0, mid - 45): mid + 45]
        if len(win) < 90:
            win = np.concatenate([win, np.zeros((90 - len(win), 18, 2))], axis=0)
        with torch.no_grad():
            logits = m1(torch.from_numpy(win[:90].reshape(1, 90, 36).astype(np.float32)))
            pred_ex = MMFIT_CLASSES[int(logits.argmax())]
        count, bounds = count_reps(pose_n, fps=VIDEO_FPS, exercise=exercise)
        set_feats = []
        for ri, (a, b) in enumerate(bounds):
            t0 = time.perf_counter()
            duration = (b - a) / VIDEO_FPS
            rom = float(np.ptp(pose_n[a:b + 1, 8, 1]))
            speed = rom / max(duration, 1e-3)
            set_feats.append([duration, rom, speed, 0.5, 0.0, 0.0, ri / max(count - 1, 1), ri * duration])
            risk = 0.0
            if m3 is not None and len(set_feats) >= 2:
                x = torch.tensor([set_feats], dtype=torch.float32)
                # pad features to in_dim with masks
                if x.shape[-1] < 10:
                    pad = torch.zeros(1, x.shape[1], 10 - x.shape[-1])
                    x = torch.cat([x, pad], dim=-1)
                with torch.no_grad():
                    _f, r = m3(x, torch.tensor([len(set_feats)]))
                    risk = float(r[0, -1])
            decision = policy.decide(risk=risk)
            dt_ms = (time.perf_counter() - t0) * 1000
            latencies.append(dt_ms)
            if decision.type != "none":
                messages.append({"set": set_id, "rep": ri, "msg": decision.message, "risk": risk, "latency_ms": dt_ms})
                voice.say(decision.message)
                print(f"set{set_id} rep{ri}: {decision.message} (risk={risk:.2f}, {dt_ms:.0f} ms)", flush=True)
    voice.close()
    import statistics
    return {
        "session": session,
        "messages": messages,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "n_reps": len(latencies),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--replay", default="w20")
    p.add_argument("--no-voice", action="store_true")
    p.add_argument("--max-sets", type=int, default=3)
    p.add_argument("--kafka", action="store_true", help="Also consume Kafka (optional)")
    args = p.parse_args(argv)
    stats = run_replay_offline(args.replay, no_voice=args.no_voice, max_sets=args.max_sets)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
