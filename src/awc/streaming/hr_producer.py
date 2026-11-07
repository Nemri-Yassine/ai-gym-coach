"""Publish watch heart rate to watch.hr.

    python -m awc.streaming.hr_producer --source replay --session w20_replay
    python -m awc.streaming.hr_producer --source replay --hr-csv path/hr.csv --session s01
    python -m awc.streaming.hr_producer --source pulsoid --token ... --session live1

Replay sources:
  - MM-Fit w20_sw_l_hr.npy (default): scheduled with the approximate w20 video↔sensor map
  - hr.csv with columns pc_ts_ms,bpm: relative timing from the first row

Live sources (Pulsoid / HypeRate / Sensor Logger) reuse awc.streaming.hr_sources.
None of them has been confirmed on the Galaxy Watch 7.
"""
from __future__ import annotations

import argparse
import csv
import threading
import time
from pathlib import Path

import numpy as np

from awc.config import KAFKA_BOOTSTRAP, REPLAY_HR_NPY, VIDEO_FPS
from awc.streaming.hr_sources import iter_hyperate, iter_pulsoid, iter_sensorlogger
from awc.streaming.kafka_io import ensure_topics, make_producer, publish
from awc.streaming.schemas import HR_TOPIC, WatchHr
from awc.streaming.w20_sync import video_frame_from_sensor


def load_hr_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((float(row["pc_ts_ms"]), float(row["bpm"])))
    return rows


def load_mmfit_hr_npy(path=REPLAY_HR_NPY):
    """Rows [sensor_frame, ts_ms, bpm] → (video_time_s, bpm) via approximate w20 map."""
    arr = np.load(path)
    out = []
    for sensor_frame, _ts, bpm in arr:
        video_frame = video_frame_from_sensor(float(sensor_frame))
        out.append((video_frame / float(VIDEO_FPS), float(bpm)))
    out.sort(key=lambda row: row[0])
    return out


def run_replay_csv(path, session, bootstrap=KAFKA_BOOTSTRAP, max_seconds=None, realtime=True):
    rows = load_hr_csv(path)
    if not rows:
        return {"session": session, "messages": 0, "seconds": 0.0, "msg_per_s": 0.0}
    t0_pc = rows[0][0]
    schedule = [((pc - t0_pc) / 1000.0, bpm) for pc, bpm in rows]
    return _run_schedule(schedule, session, bootstrap, max_seconds, realtime, source=str(path))


def run_replay_mmfit(path=REPLAY_HR_NPY, session="w20_replay", bootstrap=KAFKA_BOOTSTRAP,
                     max_seconds=None, realtime=True):
    schedule = load_mmfit_hr_npy(path)
    return _run_schedule(schedule, session, bootstrap, max_seconds, realtime, source=str(path))


def _run_schedule(schedule, session, bootstrap, max_seconds, realtime, source):
    ensure_topics(bootstrap)
    producer = make_producer(bootstrap)
    sent = 0
    t0 = time.perf_counter()
    try:
        for delay_s, bpm in schedule:
            if max_seconds is not None and delay_s > max_seconds:
                break
            if realtime:
                target = t0 + delay_s
                sleep_for = target - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
            msg = WatchHr(session=session, pc_ts_ms=int(time.time() * 1000), bpm=float(bpm))
            try:
                publish(producer, HR_TOPIC, msg)
                sent += 1
            except ValueError:
                continue
        producer.flush()
    finally:
        producer.close()
    elapsed = time.perf_counter() - t0
    return {
        "session": session,
        "source": source,
        "messages": sent,
        "seconds": elapsed,
        "msg_per_s": sent / elapsed if elapsed > 0 else 0.0,
    }


def run_live(source, session, token="", port=8081, bootstrap=KAFKA_BOOTSTRAP, max_seconds=None):
    ensure_topics(bootstrap)
    producer = make_producer(bootstrap)
    stop = threading.Event()
    sent = 0
    t0 = time.perf_counter()
    try:
        if source == "pulsoid":
            stream = iter_pulsoid(token)
        elif source == "hyperate":
            stream = iter_hyperate(token)
        elif source == "sensorlogger":
            stream = iter_sensorlogger(port, stop)
        else:
            raise ValueError(source)
        for bpm in stream:
            if max_seconds is not None and (time.perf_counter() - t0) >= max_seconds:
                break
            msg = WatchHr(session=session, pc_ts_ms=int(time.time() * 1000), bpm=float(bpm))
            try:
                publish(producer, HR_TOPIC, msg)
                sent += 1
            except ValueError:
                continue
        producer.flush()
    finally:
        stop.set()
        producer.close()
    elapsed = time.perf_counter() - t0
    return {
        "session": session,
        "source": source,
        "messages": sent,
        "seconds": elapsed,
        "msg_per_s": sent / elapsed if elapsed > 0 else 0.0,
        "watch_tested": False,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Heart rate → Kafka")
    p.add_argument("--source", required=True, choices=["pulsoid", "hyperate", "sensorlogger", "replay"])
    p.add_argument("--session", required=True)
    p.add_argument("--token", default="", help="Pulsoid token or HypeRate id (not tested on Watch 7)")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--hr-csv", default="", help="Replay an hr.csv (pc_ts_ms,bpm)")
    p.add_argument("--hr-npy", default="", help="Replay an MM-Fit sw_l_hr.npy (default: w20)")
    p.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--no-realtime", action="store_true")
    args = p.parse_args(argv)
    if args.source == "replay":
        if args.hr_csv:
            stats = run_replay_csv(
                args.hr_csv, args.session, args.bootstrap, args.max_seconds, realtime=not args.no_realtime,
            )
        else:
            path = Path(args.hr_npy) if args.hr_npy else REPLAY_HR_NPY
            stats = run_replay_mmfit(
                path, args.session, args.bootstrap, args.max_seconds, realtime=not args.no_realtime,
            )
    else:
        stats = run_live(args.source, args.session, args.token, args.port, args.bootstrap, args.max_seconds)
    print(stats, flush=True)


if __name__ == "__main__":
    main()
