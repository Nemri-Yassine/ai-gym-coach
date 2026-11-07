"""Spark job: per-rep history table + within-session personal baselines.

    python -m awc.bigdata.history
    python -m awc.bigdata.history --no-spark

Baselines are within one session and exercise: set k uses sets 1…k−1 only
(Window rowsBetween unboundedPreceding, -1). Set 1 has has_baseline = 0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from awc.config import INTERIM, MMFIT_CLASSES, MODELS, PROCESSED, VIDEO_FPS
from awc.data import mmfit
from awc.features.reps import count_reps, movement_signal, rep_boundaries
from awc.features.skeleton import normalize
from awc.features.sync import POSE_X, POSE_Y, session_split
from awc.spark import get_spark

NO_HR_SESSIONS = frozenset({"w05", "w10", "w18"})
REP_TABLE = PROCESSED / "rep_table.parquet"
BASELINES = PROCESSED / "baselines.parquet"

REP_COLUMNS = [
    "source", "user", "session", "session_date", "exercise", "set_id",
    "set_idx_in_exercise", "rep_idx", "n_reps_in_set",
    "duration_s", "rom", "mean_speed", "form_score",
    "hr_mean", "hr_end", "hr_rise", "hr_mask",
]


def _pose_from_df(df: pd.DataFrame) -> np.ndarray:
    xs = df[POSE_X].to_numpy(dtype=np.float64)
    ys = df[POSE_Y].to_numpy(dtype=np.float64)
    return np.stack([xs, ys], axis=-1)


def _rom_and_duration(pose_norm, start, end, fps=VIDEO_FPS):
    seg = pose_norm[start:end + 1]
    if len(seg) < 2:
        return np.nan, np.nan, np.nan
    sig = movement_signal(seg)
    rom = float(np.nanmax(sig) - np.nanmin(sig))
    duration_s = float((end - start) / fps)
    speed = rom / duration_s if duration_s > 0 else np.nan
    return duration_s, rom, speed


def _load_form_rf():
    path = MODELS / "m2_rf.pickle"
    if not path.is_file():
        return None
    import pickle
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _form_score_for_rep(rf_pack, exercise, pose_norm, start, end):
    if rf_pack is None:
        return np.nan
    ex_key = {"squats": "squat", "lunges": "lunge"}.get(exercise)
    if ex_key is None or ex_key not in rf_pack.get("forests", {}):
        return np.nan
    from awc.features.form_dataset import resample
    from awc.features.form_features import feature_vector, features_2d

    seg = pose_norm[start:end + 1]
    if len(seg) < 8 or not np.isfinite(seg).all():
        return np.nan
    try:
        rep64 = resample(seg, 64)
        feats = feature_vector(features_2d(rep64)).reshape(1, -1)
        forest = rf_pack["forests"][ex_key]
        proba = forest.predict_proba(feats)[0]
        classes = list(forest.classes_)
        return float(proba[classes.index(0)]) if 0 in classes else float(proba.max())
    except Exception:  # noqa: BLE001
        return np.nan


def build_rep_rows_for_session(session_id, synced_df: pd.DataFrame, rf_pack=None) -> pd.DataFrame:
    """One row per detected rep from a synced MM-Fit session DataFrame."""
    rows = []
    labels = mmfit.load_labels(session_id)
    # set_idx_in_exercise: 1.. within session
    ex_counters = {}
    set_idx_map = {}
    for set_id, (_s0, _s1, _n, exercise) in enumerate(labels):
        ex_counters[exercise] = ex_counters.get(exercise, 0) + 1
        set_idx_map[set_id] = ex_counters[exercise]

    hr_forced_off = session_id in NO_HR_SESSIONS
    for set_id, (s0, s1, n_label, exercise) in enumerate(labels):
        if exercise == "non_activity":
            continue
        block = synced_df[(synced_df["frame"] >= s0) & (synced_df["frame"] <= s1)].sort_values("frame")
        if len(block) < 16:
            continue
        pose = _pose_from_df(block)
        if not np.isfinite(pose).all():
            continue
        pose_norm = normalize(pose)
        count, bounds = count_reps(pose_norm, fps=VIDEO_FPS, exercise=exercise)
        if count == 0 or not bounds:
            continue
        hr = block["hr"].to_numpy(dtype=np.float64) if "hr" in block.columns else np.full(len(block), np.nan)
        hr_mask_col = block["hr_mask"].to_numpy() if "hr_mask" in block.columns else np.zeros(len(block))
        if hr_forced_off:
            hr_mask_set = 0
        else:
            hr_mask_set = int(np.nanmax(hr_mask_col) > 0) if len(hr_mask_col) else 0
        hr_start = float(hr[np.isfinite(hr)][0]) if np.isfinite(hr).any() else np.nan
        for rep_idx, (a, b) in enumerate(bounds):
            duration_s, rom, speed = _rom_and_duration(pose_norm, a, b)
            seg_hr = hr[a:b + 1] if b < len(hr) else hr[a:]
            seg_hr = seg_hr[np.isfinite(seg_hr)]
            hr_mean = float(np.mean(seg_hr)) if len(seg_hr) else np.nan
            hr_end = float(seg_hr[-1]) if len(seg_hr) else np.nan
            hr_rise = (hr_end - hr_start) if np.isfinite(hr_end) and np.isfinite(hr_start) else np.nan
            rows.append({
                "source": "mmfit",
                "user": session_id,  # unknown person map; session is its own history key (unused for primary baseline)
                "session": session_id,
                "session_date": session_id,  # order by set_idx, not date
                "exercise": exercise,
                "set_id": int(set_id),
                "set_idx_in_exercise": int(set_idx_map[set_id]),
                "rep_idx": int(rep_idx),
                "n_reps_in_set": int(count),
                "duration_s": duration_s,
                "rom": rom,
                "mean_speed": speed,
                "form_score": _form_score_for_rep(rf_pack, exercise, pose_norm, a, b),
                "hr_mean": hr_mean,
                "hr_end": hr_end,
                "hr_rise": hr_rise,
                "hr_mask": hr_mask_set,
            })
    return pd.DataFrame(rows, columns=REP_COLUMNS)


def build_rep_table(sessions=None, use_spark=True) -> pd.DataFrame:
    sessions = sessions or mmfit.sessions()
    rf_pack = _load_form_rf()
    frames = []
    for sid in sessions:
        path = INTERIM / "mmfit_synced" / f"session={sid}"
        if not path.is_dir():
            print(f"skip {sid}: missing {path}", flush=True)
            continue
        df = pd.read_parquet(path)
        if "session" not in df.columns:
            df = df.copy()
            df["session"] = sid
        frames.append(build_rep_rows_for_session(sid, df, rf_pack=rf_pack))
        print(f"{sid}: {len(frames[-1])} reps", flush=True)
    if not frames:
        return pd.DataFrame(columns=REP_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def add_within_session_baselines(rep_df: pd.DataFrame) -> pd.DataFrame:
    """Set-level baselines from earlier sets of the same exercise in the same session."""
    if rep_df.empty:
        return rep_df.copy()
    set_stats = (
        rep_df.sort_values(["session", "exercise", "set_id", "rep_idx"])
        .groupby(["session", "exercise", "set_id", "set_idx_in_exercise"], as_index=False)
        .agg(
            f3_duration=("duration_s", lambda s: float(s.head(3).mean())),
            f3_rom=("rom", lambda s: float(s.head(3).mean())),
            f3_speed=("mean_speed", lambda s: float(s.head(3).mean())),
            f3_hr_rise=("hr_rise", lambda s: float(s.head(3).mean())),
        )
    )
    set_stats = set_stats.sort_values(["session", "exercise", "set_idx_in_exercise"])

    def past_stats(g):
        g = g.sort_values("set_idx_in_exercise").copy()
        for col, out in (
            ("f3_duration", "base_duration"),
            ("f3_rom", "base_rom"),
            ("f3_speed", "base_speed"),
            ("f3_hr_rise", "base_hr_rise"),
        ):
            past_mean = g[col].expanding().mean().shift(1)
            past_std = g[col].expanding().std(ddof=0).shift(1)
            g[f"{out}_mean"] = past_mean
            g[f"{out}_std"] = past_std
        g["has_baseline"] = (g["set_idx_in_exercise"] > 1).astype(np.int32)
        return g

    bases = set_stats.groupby(["session", "exercise"], group_keys=False).apply(past_stats)
    return bases.reset_index(drop=True)


def add_baselines_spark(rep_df: pd.DataFrame, cores="local[2]") -> pd.DataFrame:
    """Same within-session baselines via Spark window functions."""
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    spark = get_spark("awc-history", cores=cores)
    try:
        set_stats = (
            rep_df.sort_values(["session", "exercise", "set_id", "rep_idx"])
            .groupby(["session", "exercise", "set_id", "set_idx_in_exercise"], as_index=False)
            .agg(
                f3_duration=("duration_s", lambda s: s.head(3).mean()),
                f3_rom=("rom", lambda s: s.head(3).mean()),
                f3_speed=("mean_speed", lambda s: s.head(3).mean()),
                f3_hr_rise=("hr_rise", lambda s: s.head(3).mean()),
            )
        )
        sdf = spark.createDataFrame(set_stats)
        w = (
            Window.partitionBy("session", "exercise")
            .orderBy("set_idx_in_exercise")
            .rowsBetween(Window.unboundedPreceding, -1)
        )
        for col, out in (
            ("f3_duration", "base_duration"),
            ("f3_rom", "base_rom"),
            ("f3_speed", "base_speed"),
            ("f3_hr_rise", "base_hr_rise"),
        ):
            sdf = sdf.withColumn(f"{out}_mean", F.avg(col).over(w))
            sdf = sdf.withColumn(f"{out}_std", F.stddev_pop(col).over(w))
        sdf = sdf.withColumn(
            "has_baseline",
            F.when(F.col("set_idx_in_exercise") > 1, F.lit(1)).otherwise(F.lit(0)).cast("int"),
        )
        return sdf.toPandas()
    finally:
        spark.stop()


def run(sessions=None, use_spark=True, cores="local[2]"):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    rep_df = build_rep_table(sessions=sessions, use_spark=use_spark)
    rep_df.to_parquet(REP_TABLE, index=False)
    if use_spark:
        try:
            bases = add_baselines_spark(rep_df, cores=cores)
        except Exception as exc:  # noqa: BLE001
            print(f"Spark baselines failed ({exc}); pandas fallback", flush=True)
            bases = add_within_session_baselines(rep_df)
    else:
        bases = add_within_session_baselines(rep_df)
    bases.to_parquet(BASELINES, index=False)
    summary = {
        "rep_rows": int(len(rep_df)),
        "baseline_rows": int(len(bases)),
        "sessions": sorted(rep_df["session"].unique().tolist()) if len(rep_df) else [],
        "has_baseline_rate": float(bases["has_baseline"].mean()) if len(bases) else 0.0,
    }
    (PROCESSED / "rep_table_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", nargs="*", default=None)
    p.add_argument("--no-spark", action="store_true")
    p.add_argument("--cores", default="local[2]")
    args = p.parse_args(argv)
    run(sessions=args.sessions, use_spark=not args.no_spark, cores=args.cores)


if __name__ == "__main__":
    main()
