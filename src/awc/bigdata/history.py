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

from awc.config import (
    INTERIM,
    MMFIT_CLASSES,
    MODELS,
    OWN_INTERIM,
    OWN_MANIFEST,
    OWN_REP_TABLE,
    PROCESSED,
    VIDEO_FPS,
)
from awc.data import mmfit
from awc.data.own_manifest import load_manifest
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

OWN_REP_COLUMNS = REP_COLUMNS + [
    "person", "set", "speed_rel_first2", "velocity_loss",
    "reps_truth", "segmentation_failed", "use",
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


def build_own_rep_rows(manifest_rows=None, rf_pack=None) -> pd.DataFrame:
    """One row per detected rep from data/interim/own/ using the corrected manifest."""
    if rf_pack is None:
        rf_pack = _load_form_rf()
    rows_in = manifest_rows if manifest_rows is not None else load_manifest(OWN_MANIFEST)
    rows_in = [r for r in rows_in if r["use"] in ("eval", "test")]
    out = []
    # set_idx_in_exercise per person+exercise
    counters: dict[tuple[str, str], int] = {}
    for row in rows_in:
        person = row["person"]
        session = row["session_folder"]
        set_n = int(row["set"])
        exercise = row["exercise"]
        path = OWN_INTERIM / person / f"{session}_set{set_n:02d}.parquet"
        if not path.is_file():
            print(f"skip own {person}/{session}/set{set_n:02d}: missing parquet", flush=True)
            continue
        df = pd.read_parquet(path)
        if len(df) < 16:
            continue
        pose = _pose_from_df(df)
        good = np.isfinite(pose).all(axis=(1, 2))
        if good.sum() < 16:
            continue
        # keep timeline but drop all-NaN frames for counting
        pose_use = pose[good]
        pose_norm = normalize(pose_use)
        count, bounds = count_reps(pose_norm, fps=VIDEO_FPS, exercise=exercise)
        if count == 0 or not bounds:
            continue
        truth = row.get("reps")
        try:
            truth_n = int(truth) if truth not in (None, "") else None
        except (TypeError, ValueError):
            truth_n = None
        # Implausible counts vs ground truth → segmentation_failed (exclude from velocity medians)
        seg_failed = False
        if truth_n is not None and truth_n > 0:
            if count < 0.5 * truth_n or count > 1.5 * truth_n:
                seg_failed = True
        key = (person, exercise)
        counters[key] = counters.get(key, 0) + 1
        set_idx = counters[key]
        hr = df["hr"].to_numpy(dtype=np.float64) if "hr" in df.columns else np.full(len(df), np.nan)
        hr_mask_col = df["hr_mask"].to_numpy() if "hr_mask" in df.columns else np.zeros(len(df))
        # map good-frame indices back onto full HR if needed: use good mask
        hr_g = hr[good] if len(hr) == len(pose) else hr[: len(pose_use)]
        hr_m = hr_mask_col[good] if len(hr_mask_col) == len(pose) else hr_mask_col[: len(pose_use)]
        hr_mask_set = int(np.nanmax(hr_m) > 0) if len(hr_m) else 0
        hr_start = float(hr_g[np.isfinite(hr_g)][0]) if np.isfinite(hr_g).any() else np.nan
        session_date = session.split("_")[0] if "_" in session else session
        set_reps = []
        for rep_idx, (a, b) in enumerate(bounds):
            duration_s, rom, speed = _rom_and_duration(pose_norm, a, b)
            seg_hr = hr_g[a:b + 1]
            seg_hr = seg_hr[np.isfinite(seg_hr)]
            hr_mean = float(np.mean(seg_hr)) if len(seg_hr) else np.nan
            hr_end = float(seg_hr[-1]) if len(seg_hr) else np.nan
            hr_rise = (hr_end - hr_start) if np.isfinite(hr_end) and np.isfinite(hr_start) else np.nan
            set_reps.append({
                "source": "own",
                "user": person,
                "session": f"{person}/{session}",
                "session_date": session_date,
                "exercise": exercise,
                "set_id": set_n,
                "set_idx_in_exercise": int(set_idx),
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
                "person": person,
                "set": set_n,
                "speed_rel_first2": np.nan,
                "velocity_loss": np.nan,
                "reps_truth": truth_n if truth_n is not None else np.nan,
                "segmentation_failed": bool(seg_failed),
                "use": row.get("use") or "eval",
            })
        # within-set: speed relative to mean of first 2 reps; velocity loss = 1 - rel
        speeds = [r["mean_speed"] for r in set_reps]
        base = float(np.nanmean(speeds[:2])) if speeds else np.nan
        for r in set_reps:
            if seg_failed:
                r["speed_rel_first2"] = np.nan
                r["velocity_loss"] = np.nan
            elif np.isfinite(base) and base > 1e-8 and np.isfinite(r["mean_speed"]):
                rel = float(r["mean_speed"] / base)
                r["speed_rel_first2"] = rel
                r["velocity_loss"] = float(1.0 - rel)
        out.extend(set_reps)
        flag = " segmentation_failed" if seg_failed else ""
        print(f"own {person}/{session}/set{set_n:02d}: {count} reps{flag}", flush=True)
    return pd.DataFrame(out, columns=OWN_REP_COLUMNS)


def add_own_within_set_spark(rep_df: pd.DataFrame, cores="local[2]") -> pd.DataFrame:
    """Recompute speed_rel_first2 / velocity_loss with Spark window functions (sanity)."""
    if rep_df.empty:
        return rep_df.copy()
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    spark = get_spark("awc-own-history", cores=cores)
    try:
        sdf = spark.createDataFrame(rep_df.drop(columns=["speed_rel_first2", "velocity_loss"], errors="ignore"))
        w_first2 = (
            Window.partitionBy("person", "session", "set")
            .orderBy("rep_idx")
            .rowsBetween(Window.unboundedPreceding, 1)
        )
        # mean of first two rows' speeds as baseline for the whole set
        sdf = sdf.withColumn("_base", F.avg("mean_speed").over(w_first2))
        # freeze baseline as the value on rep_idx <= 1, then broadcast via max over set
        w_set = Window.partitionBy("person", "session", "set")
        sdf = sdf.withColumn(
            "_base_set",
            F.max(F.when(F.col("rep_idx") <= 1, F.col("_base")).otherwise(F.lit(None))).over(w_set),
        )
        sdf = sdf.withColumn(
            "speed_rel_first2",
            F.when(F.col("_base_set") > 0, F.col("mean_speed") / F.col("_base_set")).otherwise(F.lit(None)),
        )
        sdf = sdf.withColumn("velocity_loss", F.lit(1.0) - F.col("speed_rel_first2"))
        sdf = sdf.drop("_base", "_base_set")
        return sdf.toPandas()
    finally:
        spark.stop()


def run_own(use_spark=True, cores="local[2]") -> dict:
    """Build data/interim/own_rep_table.parquet from own interim pose."""
    OWN_INTERIM.mkdir(parents=True, exist_ok=True)
    rep_df = build_own_rep_rows()
    if use_spark and not rep_df.empty:
        try:
            rep_df = add_own_within_set_spark(rep_df, cores=cores)
        except Exception as exc:  # noqa: BLE001
            print(f"Spark own within-set failed ({exc}); keeping pandas values", flush=True)
    OWN_REP_TABLE.parent.mkdir(parents=True, exist_ok=True)
    rep_df.to_parquet(OWN_REP_TABLE, index=False)
    # median velocity loss at the last rep per exercise (eval only; skip segmentation_failed)
    last_loss = {}
    n_seg_failed = 0
    n_eval_sets = 0
    n_test_sets = 0
    if not rep_df.empty:
        if "segmentation_failed" in rep_df.columns:
            n_seg_failed = int(
                rep_df.groupby(["person", "session", "set"])["segmentation_failed"].any().sum()
            )
        if "use" in rep_df.columns:
            n_eval_sets = int(rep_df[rep_df["use"] == "eval"].groupby(["person", "session", "set"]).ngroups)
            n_test_sets = int(rep_df[rep_df["use"] == "test"].groupby(["person", "session", "set"]).ngroups)
        usable = rep_df
        if "use" in usable.columns:
            usable = usable[usable["use"] == "eval"]
        if "segmentation_failed" in usable.columns:
            usable = usable[~usable["segmentation_failed"].astype(bool)]
        if not usable.empty:
            last = usable.sort_values("rep_idx").groupby(
                ["person", "session", "set", "exercise"], as_index=False
            ).tail(1)
            for ex, g in last.groupby("exercise"):
                vals = g["velocity_loss"].to_numpy(dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    last_loss[ex] = round(float(np.nanmedian(vals)), 3)
    summary = {
        "rep_rows": int(len(rep_df)),
        "sets": int(rep_df.groupby(["person", "session", "set"]).ngroups) if len(rep_df) else 0,
        "eval_sets": n_eval_sets,
        "test_sets": n_test_sets,
        "segmentation_failed_sets": n_seg_failed,
        "people": sorted(rep_df["person"].unique().tolist()) if len(rep_df) else [],
        "median_velocity_loss_last_rep": last_loss,
        "path": str(OWN_REP_TABLE),
    }
    (INTERIM / "own_rep_table_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


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
    p.add_argument("--own", action="store_true", help="build own_rep_table.parquet from interim/own")
    args = p.parse_args(argv)
    if args.own:
        run_own(use_spark=not args.no_spark, cores=args.cores)
    else:
        run(sessions=args.sessions, use_spark=not args.no_spark, cores=args.cores)


if __name__ == "__main__":
    main()
