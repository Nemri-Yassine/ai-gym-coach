"""Spark job: one synced Parquet partition per MM-Fit session.

    python -m awc.bigdata.build_interim --cores local[4]
    python -m awc.bigdata.build_interim --sessions w00 --no-spark
    python -m awc.bigdata.build_interim --scaling
"""
import argparse
import json
import subprocess
import sys
import time

import pandas as pd

from awc.config import INTERIM, MMFIT_SPLIT
from awc.data import mmfit
from awc.features.sync import SYNC_COLUMNS, sync_session
from awc.training.tracking import new_run, save_metrics


def _summary(df, runtime_s, cores):
    per_session = {}
    for session, g in df.groupby("session"):
        per_session[str(session)] = {
            "rows": int(len(g)),
            "pct_imu": float(g["acc_x"].notna().mean()),
            "pct_hr": float((g["hr_mask"] == 1).mean()),
        }
    return {
        "rows_per_session": per_session,
        "n_rows": int(len(df)),
        "n_sessions": int(df["session"].nunique()),
        "pct_frames_with_imu": float(df["acc_x"].notna().mean()),
        "pct_frames_with_hr": float((df["hr_mask"] == 1).mean()),
        "rows_per_class": {str(k): int(v) for k, v in df["exercise"].value_counts().items()},
        "runtime_s": round(float(runtime_s), 2),
        "cores": cores,
    }


def _write_summary(payload):
    INTERIM.mkdir(parents=True, exist_ok=True)
    path = INTERIM / "mmfit_synced_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("n_rows", "n_sessions", "pct_frames_with_imu", "pct_frames_with_hr", "runtime_s", "cores")}, indent=2))
    return path


def _spark_schema():
    from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, StructField, StructType
    fields = [StructField("frame", LongType())]
    for col in SYNC_COLUMNS[1:]:
        if col in {"hr_mask", "set_id", "set_reps"}:
            fields.append(StructField(col, IntegerType()))
        elif col in {"exercise", "session", "split"}:
            fields.append(StructField(col, StringType()))
        else:
            fields.append(StructField(col, DoubleType()))
    return StructType(fields)


def _sync_pandas(pdf: pd.DataFrame) -> pd.DataFrame:
    session = str(pdf["session"].iloc[0])
    print(f"sync {session}", flush=True)
    return sync_session(session)


def build(sessions, cores="local[4]", use_spark=True):
    t0 = time.perf_counter()
    if use_spark:
        from awc.spark import get_spark, spark_path, stop_spark
        spark = get_spark("mmfit_sync", cores)
        spark.sparkContext.setLogLevel("WARN")
        spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
        # The session list is tiny, so adaptive coalescing would put every
        # session in one task and extra cores would do nothing. One partition
        # per session lets local[n] actually run n sessions at once.
        spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "false")
        spark.conf.set("spark.sql.shuffle.partitions", str(len(sessions)))
        src = spark.createDataFrame([(s,) for s in sessions], ["session"])
        src = src.repartition(len(sessions), "session")
        out = src.groupBy("session").applyInPandas(_sync_pandas, _spark_schema())
        dest = spark_path(INTERIM / "mmfit_synced")
        out.write.mode("overwrite").partitionBy("session").parquet(dest)
        df = pd.read_parquet(INTERIM / "mmfit_synced")
        stop_spark()
    else:
        frames = []
        for session in sessions:
            print(f"sync {session}", flush=True)
            frames.append(sync_session(session))
        df = pd.concat(frames, ignore_index=True)
        dest = INTERIM / "mmfit_synced"
        df.to_parquet(dest, partition_cols=["session"], index=False)
    runtime = time.perf_counter() - t0
    payload = _summary(df, runtime, "pandas" if not use_spark else cores)
    _write_summary(payload)
    return payload


def run_scaling(sessions):
    """Time the same job on local[1], local[2] and local[4]."""
    run_dir = new_run("spark_scaling", {"sessions": list(sessions), "cores": ["local[1]", "local[2]", "local[4]"]})
    timings = {}
    for cores in ("local[1]", "local[2]", "local[4]"):
        print(f"=== scaling {cores} ===", flush=True)
        t0 = time.perf_counter()
        subprocess.check_call([sys.executable, "-m", "awc.bigdata.build_interim", "--cores", cores, "--sessions", *sessions])
        timings[cores] = round(time.perf_counter() - t0, 2)
        print(f"{cores} {timings[cores]} s", flush=True)
    save_metrics(run_dir, {"runtime_s": timings, "n_sessions": len(sessions)})
    print(f"RUN_DIR={run_dir}")
    return timings


def default_sessions():
    known = [s for ids in MMFIT_SPLIT.values() for s in ids]
    on_disk = set(mmfit.sessions())
    return [s for s in known if s in on_disk]


def main(argv=None):
    p = argparse.ArgumentParser(description="Sync MM-Fit streams to Parquet")
    p.add_argument("--cores", default="local[4]")
    p.add_argument("--no-spark", action="store_true")
    p.add_argument("--sessions", nargs="*")
    p.add_argument("--scaling", action="store_true")
    args = p.parse_args(argv)
    sessions = args.sessions or default_sessions()
    if args.scaling:
        run_scaling(sessions)
        return
    build(sessions, cores=args.cores, use_spark=not args.no_spark)


if __name__ == "__main__":
    main()
