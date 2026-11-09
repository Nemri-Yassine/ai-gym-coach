"""Optional after-session import: Sensor Logger (Wear OS) CSV → imu.csv aligned by Clap events.

    python scripts/import_watch_export.py <session_folder> <sensor_logger.csv>

The watch IMU is not streamed live by any free app without custom code. Export after the
session and align with clap markers in events.csv (event=clap).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description="Import Sensor Logger export into a session folder")
    p.add_argument("session_folder")
    p.add_argument("csv_path")
    p.add_argument("--clap-offset-ms", type=float, default=0.0, help="watch_ts - pc_ts at first clap")
    args = p.parse_args(argv)
    folder = Path(args.session_folder)
    events = folder / "events.csv"
    claps = []
    if events.is_file():
        with events.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("event") == "clap":
                    claps.append(int(float(row["pc_ts_ms"])))
    out = folder / "imu.csv"
    # Pass-through copy with offset note; full axis mapping depends on Sensor Logger columns.
    n = 0
    with open(args.csv_path, newline="", encoding="utf-8") as src, out.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        fieldnames = ["pc_ts_ms"] + [c for c in (reader.fieldnames or []) if c]
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            n += 1
            # Prefer a time column if present
            t = None
            for key in ("time", "timestamp", "Timestamp", "pc_ts_ms"):
                if key in row and row[key] not in (None, ""):
                    t = float(row[key])
                    break
            if t is None:
                continue
            # Sensor Logger often uses seconds; convert heuristically
            if t < 1e11:
                t_ms = t * 1000.0
            else:
                t_ms = t
            pc = t_ms - float(args.clap_offset_ms)
            row_out = {"pc_ts_ms": int(pc), **row}
            writer.writerow(row_out)
    print(f"wrote {out} ({n} rows); claps in session: {len(claps)}")
    if claps:
        print(f"  first clap pc_ts_ms={claps[0]} — tune --clap-offset-ms so watch time matches")


if __name__ == "__main__":
    main()
