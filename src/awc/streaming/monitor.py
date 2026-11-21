"""Print Kafka topic rates and median latency every 5 seconds.

    python -m awc.streaming.monitor --seconds 120
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict

from awc.config import KAFKA_BOOTSTRAP, KAFKA_TOPICS
from awc.streaming.kafka_io import ensure_topics, make_consumer


def run(bootstrap=KAFKA_BOOTSTRAP, seconds=120, interval=5.0, group_id=None):
    ensure_topics(bootstrap)
    group_id = group_id or f"awc-monitor-{int(time.time())}"
    consumer = make_consumer(group_id, KAFKA_TOPICS, bootstrap=bootstrap, from_beginning=False)
    counts = defaultdict(int)
    latencies = defaultdict(list)
    t0 = time.perf_counter()
    next_report = t0 + interval
    reports = []
    try:
        while time.perf_counter() - t0 < seconds:
            polled = consumer.poll(timeout_ms=200)
            now_ms = time.time() * 1000.0
            for _tp, records in polled.items():
                for record in records:
                    topic = record.topic
                    counts[topic] += 1
                    try:
                        payload = json.loads(record.value)
                        pc = payload.get("pc_ts_ms")
                        if pc is not None:
                            latencies[topic].append(now_ms - float(pc))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
            if time.perf_counter() >= next_report:
                elapsed = time.perf_counter() - (next_report - interval)
                row = {"t": round(time.perf_counter() - t0, 1)}
                for topic in KAFKA_TOPICS:
                    n = counts[topic]
                    rates = n / max(elapsed, 1e-6)
                    med = statistics.median(latencies[topic]) if latencies[topic] else None
                    row[topic] = {"msg_s": round(rates, 2), "n": n, "median_latency_ms": None if med is None else round(med, 1)}
                    counts[topic] = 0
                    latencies[topic].clear()
                reports.append(row)
                print(json.dumps(row), flush=True)
                next_report += interval
    finally:
        consumer.close()
    return reports


def main(argv=None):
    p = argparse.ArgumentParser(description="Kafka throughput / latency monitor")
    p.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--interval", type=float, default=5)
    args = p.parse_args(argv)
    run(args.bootstrap, args.seconds, args.interval)


if __name__ == "__main__":
    main()
