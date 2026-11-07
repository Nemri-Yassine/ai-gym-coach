"""Model 3 sequences: one sample = one set = sequence of reps (MM-Fit only).

Normalisations: general (global train z-score) vs personal (within-session baseline
from step 2.5, fallback to global when has_baseline = 0).

Labels: fatigue = rep_idx/(n-1); breakdown = speed drop ≥ threshold vs first-2-rep
mean or form_score < 0.5; risk = breakdown at this rep or within the next 2.
Features at rep t use only reps ≤ t (causal).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from awc.config import MMFIT_SPLIT, PROCESSED

REP_TABLE = PROCESSED / "rep_table.parquet"
BASELINES = PROCESSED / "baselines.parquet"
M3_DIR = PROCESSED / "m3"
FEATURE_COLS = [
    "duration_s", "rom", "mean_speed", "form_score", "hr_end", "hr_rise",
    "rep_frac", "time_since_start",
]
MASK_COLS = ["form_mask", "hr_feat_mask"]


def _split_of(session: str) -> str:
    for name, ids in MMFIT_SPLIT.items():
        if session in ids:
            return name
    return "unused"


def breakdown_index(speeds, form_scores=None, speed_drop=0.20, form_thresh=0.5):
    """First rep index where speed dropped ≥ speed_drop vs mean of first 2, or form_score < thresh."""
    speeds = np.asarray(speeds, dtype=np.float64)
    n = len(speeds)
    if n < 3:
        return None
    base = np.nanmean(speeds[:2])
    if not np.isfinite(base) or base <= 0:
        return None
    form_scores = np.asarray(form_scores if form_scores is not None else [np.nan] * n, dtype=np.float64)
    for i in range(2, n):
        speed_hit = np.isfinite(speeds[i]) and speeds[i] <= (1.0 - speed_drop) * base
        form_hit = np.isfinite(form_scores[i]) and form_scores[i] < form_thresh
        if speed_hit or form_hit:
            return int(i)
    return None


def risk_labels(n, breakdown_i, horizon=2):
    """risk[t] = 1 if breakdown at t .. t+horizon."""
    risk = np.zeros(n, dtype=np.float32)
    if breakdown_i is None:
        return risk
    for t in range(n):
        if breakdown_i <= t + horizon and breakdown_i >= t:
            risk[t] = 1.0
    return risk


def fatigue_labels(n):
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    return (np.arange(n, dtype=np.float32) / float(n - 1))


def build_set_records(rep_df: pd.DataFrame, base_df: pd.DataFrame, speed_drop=0.20):
    """List of set dicts with raw features and labels (before normalisation)."""
    base_key = base_df.set_index(["session", "exercise", "set_id"])
    records = []
    for (session, exercise, set_id), g in rep_df.groupby(["session", "exercise", "set_id"]):
        g = g.sort_values("rep_idx")
        n = len(g)
        if n < 2:
            continue
        duration = g["duration_s"].to_numpy(dtype=np.float64)
        rom = g["rom"].to_numpy(dtype=np.float64)
        speed = g["mean_speed"].to_numpy(dtype=np.float64)
        form = g["form_score"].to_numpy(dtype=np.float64)
        hr_end = g["hr_end"].to_numpy(dtype=np.float64)
        hr_rise = g["hr_rise"].to_numpy(dtype=np.float64)
        hr_mask = g["hr_mask"].to_numpy(dtype=np.float64)
        time_since = np.cumsum(duration) - duration
        rep_frac = np.arange(n, dtype=np.float64) / max(n - 1, 1)
        bd = breakdown_index(speed, form, speed_drop=speed_drop)
        feats = np.column_stack([
            duration, rom, speed, form, hr_end, hr_rise, rep_frac, time_since,
        ]).astype(np.float32)
        form_mask = np.isfinite(form).astype(np.float32)
        hr_feat_mask = (hr_mask > 0).astype(np.float32)
        masks = np.column_stack([form_mask, hr_feat_mask]).astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0)
        try:
            brow = base_key.loc[(session, exercise, set_id)]
            if isinstance(brow, pd.DataFrame):
                brow = brow.iloc[0]
            has_baseline = int(brow.get("has_baseline", 0))
            personal = {
                "duration_s": (brow.get("base_duration_mean"), brow.get("base_duration_std")),
                "rom": (brow.get("base_rom_mean"), brow.get("base_rom_std")),
                "mean_speed": (brow.get("base_speed_mean"), brow.get("base_speed_std")),
                "hr_rise": (brow.get("base_hr_rise_mean"), brow.get("base_hr_rise_std")),
            }
        except KeyError:
            has_baseline = 0
            personal = {}
        records.append({
            "session": session,
            "exercise": exercise,
            "set_id": int(set_id),
            "set_idx_in_exercise": int(g["set_idx_in_exercise"].iloc[0]),
            "split": _split_of(session),
            "X": feats,
            "masks": masks,
            "fatigue": fatigue_labels(n),
            "risk": risk_labels(n, bd),
            "breakdown_i": bd,
            "has_baseline": has_baseline,
            "personal": personal,
            "raw_speed": speed.astype(np.float32),
            "n": n,
        })
    return records


def _global_stats(records, feature_idx):
    """Mean/std over all train reps for selected feature columns."""
    chunks = [r["X"][:, feature_idx] for r in records if r["split"] == "train"]
    if not chunks:
        return np.zeros(len(feature_idx)), np.ones(len(feature_idx))
    flat = np.concatenate(chunks, axis=0)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_records(records, mode="general", stats=None):
    """Return new list with X normalised. mode in {general, personal}."""
    # indices in FEATURE_COLS
    idx = {name: i for i, name in enumerate(FEATURE_COLS)}
    z_cols = ["duration_s", "rom", "mean_speed", "form_score", "hr_end", "hr_rise"]
    z_idx = [idx[c] for c in z_cols]
    if stats is None:
        mean, std = _global_stats(records, z_idx)
        stats = {"mean": mean.tolist(), "std": std.tolist(), "cols": z_cols}
    else:
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)

    out = []
    for r in records:
        x = r["X"].copy()
        if mode == "general" or r["has_baseline"] == 0:
            x[:, z_idx] = (x[:, z_idx] - mean) / std
        else:
            # personal for duration/rom/speed/hr_rise; form/hr_end stay global
            pers_map = {
                "duration_s": "duration_s",
                "rom": "rom",
                "mean_speed": "mean_speed",
                "hr_rise": "hr_rise",
            }
            for j, col in enumerate(z_cols):
                if col in pers_map and col in r["personal"]:
                    mu, sg = r["personal"][col]
                    if mu is not None and np.isfinite(mu):
                        s = sg if sg is not None and np.isfinite(sg) and sg > 1e-6 else 1.0
                        x[:, idx[col]] = (x[:, idx[col]] - float(mu)) / float(s)
                        continue
                x[:, idx[col]] = (x[:, idx[col]] - mean[j]) / std[j]
        nr = dict(r)
        nr["X"] = x.astype(np.float32)
        nr["norm"] = mode
        out.append(nr)
    return out, stats


def save_dataset(records, stats, out_dir=M3_DIR, name="general"):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sets_{name}.npz"
    # ragged: store as object arrays
    payload = {
        "session": np.array([r["session"] for r in records], dtype=object),
        "exercise": np.array([r["exercise"] for r in records], dtype=object),
        "set_id": np.array([r["set_id"] for r in records], dtype=np.int32),
        "set_idx_in_exercise": np.array([r["set_idx_in_exercise"] for r in records], dtype=np.int32),
        "split": np.array([r["split"] for r in records], dtype=object),
        "has_baseline": np.array([r["has_baseline"] for r in records], dtype=np.int32),
        "breakdown_i": np.array(
            [-1 if r["breakdown_i"] is None else r["breakdown_i"] for r in records], dtype=np.int32,
        ),
        "X": np.array([r["X"] for r in records], dtype=object),
        "masks": np.array([r["masks"] for r in records], dtype=object),
        "fatigue": np.array([r["fatigue"] for r in records], dtype=object),
        "risk": np.array([r["risk"] for r in records], dtype=object),
        "raw_speed": np.array([r["raw_speed"] for r in records], dtype=object),
        "feature_cols": np.array(FEATURE_COLS),
    }
    np.savez_compressed(path, **payload)
    (out_dir / f"stats_{name}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    counts = {s: int((payload["split"] == s).sum()) for s in ("train", "val", "test_seen", "test_unseen", "unused")}
    (out_dir / f"counts_{name}.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return path, counts


def build_all(speed_drop=0.20):
    rep_df = pd.read_parquet(REP_TABLE)
    base_df = pd.read_parquet(BASELINES)
    raw = build_set_records(rep_df, base_df, speed_drop=speed_drop)
    general, stats = normalize_records(raw, mode="general")
    personal, _ = normalize_records(raw, mode="personal", stats=stats)
    p1, c1 = save_dataset(general, stats, name="general")
    p2, c2 = save_dataset(personal, stats, name="personal")
    return {"general": (str(p1), c1), "personal": (str(p2), c2), "n_raw": len(raw)}


def features_are_causal(record) -> bool:
    """Sanity: X[t] must not depend on future reps (rep_frac and time_since are non-decreasing)."""
    x = record["X"]
    rep_frac = x[:, FEATURE_COLS.index("rep_frac")]
    tss = x[:, FEATURE_COLS.index("time_since_start")]
    return bool(np.all(np.diff(rep_frac) >= -1e-6) and np.all(np.diff(tss) >= -1e-6))
