"""Measure honest post-set latency (MediaPipe included) on 3 own sets.

    python scripts/measure_live_latency.py

Writes experiments/<date>_station_latency/metrics.json and prints pose/M1/reps/M2 timings.
Compares full MediaPipe vs live_fast (stride 2 + 480px) accuracy on the same sets.
"""
from __future__ import annotations

import json
from pathlib import Path

from awc.config import OWN_DIR, OWN_INTERIM
from awc.station.analyze import analyze_set
from awc.training.tracking import new_run, save_metrics

# Three short-ish eval sets with cached pose available for accuracy comparison.
SETS = [
    ("yassine", "2026-11-10_SQUAT", 1),
    ("partner", "2026-11-10_PUSHUP", 1),
    ("partner", "2026-11-10_s01", 1),
]


def main():
    rows = []
    for person, folder, set_n in SETS:
        session = OWN_DIR / person / folder
        cached = OWN_INTERIM / person / f"{folder}_set{set_n:02d}.parquet"
        if not session.is_dir() or not (session / "sets" / f"set{set_n:02d}.mp4").is_file():
            print(f"skip missing {person}/{folder}/set{set_n:02d}")
            continue
        cached_r = analyze_set(session, set_n, person=person, session_folder=folder, use_cache=True)
        full = analyze_set(
            session, set_n, person=person, session_folder=folder,
            use_cache=False, live_fast=False, show_form=False,
        )
        fast = analyze_set(
            session, set_n, person=person, session_folder=folder,
            use_cache=False, live_fast=True, show_form=False,
        )
        row = {
            "key": f"{person}/{folder}/set{set_n:02d}",
            "has_cache": cached.is_file(),
            "cached": {
                "exercise": cached_r["exercise"], "reps": cached_r["reps"],
                "elapsed_s": cached_r["elapsed_s"], "timings": cached_r["timings"],
            },
            "full_mediapipe": {
                "exercise": full["exercise"], "reps": full["reps"],
                "elapsed_s": full["elapsed_s"], "timings": full["timings"],
            },
            "live_fast": {
                "exercise": fast["exercise"], "reps": fast["reps"],
                "elapsed_s": fast["elapsed_s"], "timings": fast["timings"],
            },
            "exercise_match_fast_vs_cache": fast["exercise"] == cached_r["exercise"],
            "reps_delta_fast_vs_cache": int(fast["reps"]) - int(cached_r["reps"]),
        }
        rows.append(row)
        print(
            f"{row['key']}: cached {cached_r['elapsed_s']:.2f}s | "
            f"full {full['elapsed_s']:.2f}s (pose {full['timings']['pose_s']}) | "
            f"live_fast {fast['elapsed_s']:.2f}s (pose {fast['timings']['pose_s']}) | "
            f"ex {cached_r['exercise']}→{fast['exercise']} reps {cached_r['reps']}→{fast['reps']}"
        )

    run = new_run("station_latency", {"n": len(rows), "sets": [r["key"] for r in rows]})
    save_metrics(run, {
        "sets": rows,
        "mean_cached_s": _mean([r["cached"]["elapsed_s"] for r in rows]),
        "mean_full_s": _mean([r["full_mediapipe"]["elapsed_s"] for r in rows]),
        "mean_live_fast_s": _mean([r["live_fast"]["elapsed_s"] for r in rows]),
        "mean_full_pose_s": _mean([r["full_mediapipe"]["timings"]["pose_s"] for r in rows]),
        "mean_live_fast_pose_s": _mean([r["live_fast"]["timings"]["pose_s"] for r in rows]),
        "n_exercise_match_fast": sum(1 for r in rows if r["exercise_match_fast_vs_cache"]),
        "docker_kafka": False,
        "docker_kafka_note": "Docker daemon not running; station --kafka not measured",
    })
    print(f"RUN_DIR={run}")


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else None


if __name__ == "__main__":
    main()
