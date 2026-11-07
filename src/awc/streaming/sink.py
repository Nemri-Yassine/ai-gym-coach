"""Append Kafka streams to Parquet.

Real sessions (default):
  data/raw/own_recordings/<user>/<session>/stream/

Replay sessions (--replay):
  data/interim/replay/<session>/stream/

Replay must never write into own_recordings.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from awc.config import KAFKA_BOOTSTRAP, KAFKA_TOPICS, OWN_DIR, REPLAY_DIR
from awc.streaming.kafka_io import ensure_topics, make_consumer


def stream_dir(session, user=None, replay=False) -> Path:
    if replay:
        return REPLAY_DIR / session / "stream"
    if not user:
        raise ValueError("user is required for real sessions")
    return OWN_DIR / user / session / "stream"


def _append_rows(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(path, index=False)


def run(session, user=None, replay=False, bootstrap=KAFKA_BOOTSTRAP, seconds=120, group_id=None):
    out = stream_dir(session, user=user, replay=replay)
    if replay and OWN_DIR in out.resolve().parents:
        raise RuntimeError("replay sink must not write under own_recordings")
    ensure_topics(bootstrap)
    group_id = group_id or f"awc-sink-{session}-{int(time.time())}"
    consumer = make_consumer(group_id, KAFKA_TOPICS, bootstrap=bootstrap, from_beginning=False)
    buffers = {topic: [] for topic in KAFKA_TOPICS}
    t0 = time.perf_counter()
    written = {topic: 0 for topic in KAFKA_TOPICS}
    try:
        while time.perf_counter() - t0 < seconds:
            polled = consumer.poll(timeout_ms=200)
            for _tp, records in polled.items():
                for record in records:
                    try:
                        payload = json.loads(record.value)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("session") not in (None, session):
                        continue
                    buffers[record.topic].append(payload)
            for topic, rows in list(buffers.items()):
                if len(rows) >= 20 or (rows and time.perf_counter() - t0 > seconds - 0.5):
                    path = out / f"{topic.replace('.', '_')}.parquet"
                    _append_rows(path, rows)
                    written[topic] += len(rows)
                    buffers[topic] = []
        for topic, rows in buffers.items():
            if rows:
                path = out / f"{topic.replace('.', '_')}.parquet"
                _append_rows(path, rows)
                written[topic] += len(rows)
    finally:
        consumer.close()
    return {"dir": str(out), "written": written, "replay": replay}


def main(argv=None):
    p = argparse.ArgumentParser(description="Kafka → Parquet sink")
    p.add_argument("--session", required=True)
    p.add_argument("--user", default="", help="Required for real sessions")
    p.add_argument("--replay", action="store_true", help="Write under data/interim/replay/")
    p.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP)
    p.add_argument("--seconds", type=float, default=120)
    args = p.parse_args(argv)
    if not args.replay and not args.user:
        p.error("--user is required unless --replay is set")
    stats = run(args.session, args.user or None, args.replay, args.bootstrap, args.seconds)
    print(stats, flush=True)


if __name__ == "__main__":
    main()
