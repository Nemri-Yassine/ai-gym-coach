"""RQ4 on our own gym-station videos: Model 1, reps, Model 2, HR.

    python scripts/eval_own_recordings.py
    python scripts/eval_own_recordings.py --session yassine/2026-11-10_TRICEPS

Uses cached pose in data/interim/own/. Own recordings are test data only — no tuning.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from awc.config import (
    FIGURES,
    INTERIM,
    MMFIT_CLASSES,
    MODELS,
    OWN_DIR,
    OWN_INTERIM,
    OWN_MANIFEST,
    OWN_REPS_VERIFIED,
    REPORTS,
    VIDEO_FPS,
)
from awc.data.own_manifest import load_manifest
from awc.evaluation.metrics import macro_f1, plot_confusion, rep_count_errors
from awc.features.form_dataset import resample
from awc.features.form_features import feature_vector, features_2d
from awc.features.reps import count_reps, movement_signal, rep_boundaries
from awc.features.skeleton import normalize
from awc.models.m1_recognition import M1Model
from awc.training.tracking import new_run, save_metrics

WINDOW, HOP = 90, 45
CHECKPOINTS = ("m1_transformer_pose", "m1_cnn_pose", "m1_lstm_pose", "m1_transformer_both")
POSE_X = [f"x{i}" for i in range(18)]
POSE_Y = [f"y{i}" for i in range(18)]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def load_set_pose(row: dict) -> tuple[np.ndarray, pd.DataFrame]:
    path = OWN_INTERIM / row["person"] / f"{row['session_folder']}_set{int(row['set']):02d}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    xs = df[POSE_X].to_numpy(dtype=np.float64)
    ys = df[POSE_Y].to_numpy(dtype=np.float64)
    pose = np.stack([xs, ys], axis=-1)
    return pose, df


def activity_span(pose: np.ndarray, fps: int = VIDEO_FPS) -> tuple[int, int]:
    """Trim still ends via movement energy (same role as action_span on labelled bounds)."""
    n = len(pose)
    if n < fps:
        return 0, n
    good = np.isfinite(pose).all(axis=(1, 2))
    if good.sum() < fps:
        trim = int(2 * fps)
        return trim, max(trim + 1, n - trim)
    filled = pose.copy()
    if (~good).any():
        idx = np.arange(n)
        for j in range(18):
            for c in range(2):
                col = filled[:, j, c]
                if np.isfinite(col).any():
                    col[~np.isfinite(col)] = np.interp(
                        idx[~np.isfinite(col)], idx[np.isfinite(col)], col[np.isfinite(col)]
                    )
    sig = movement_signal(filled)
    energy = np.abs(np.diff(sig, prepend=sig[0]))
    thresh = float(np.median(energy) + 0.35 * np.std(energy))
    active = np.where(energy > thresh)[0]
    if len(active) < 5:
        trim = int(2 * fps)
        return trim, max(trim + 1, n - trim)
    # keep a little margin; cap trim at ~3 s
    max_trim = int(3 * fps)
    start = int(active[0])
    end = int(active[-1]) + 1
    start = min(start, max_trim)
    end = max(end, n - max_trim)
    start = max(0, start)
    end = min(n, max(start + 1, end))
    return start, end


def windows(pose, start, end):
    out = []
    for s in range(0, len(pose) - WINDOW + 1, HOP):
        centre = s + WINDOW // 2
        if not start <= centre < end:
            continue
        w = pose[s:s + WINDOW]
        if np.isfinite(w).all():
            out.append(normalize(w).reshape(WINDOW, 36))
    return np.asarray(out, dtype=np.float32).reshape(-1, WINDOW, 36)


def load_model(name: str):
    ckpt = torch.load(MODELS / f"{name}.pt", map_location="cpu", weights_only=False)
    model = M1Model(ckpt["arch"], ckpt["inputs"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def predict(model, x):
    if len(x) == 0:
        return np.zeros(0, dtype=np.int64)
    pose = torch.from_numpy(x)
    with torch.no_grad():
        if model.inputs == "pose":
            logits = model(pose)
        else:
            logits = model(pose, torch.zeros(len(x), WINDOW, 6), torch.zeros(len(x)))
    return logits.argmax(1).numpy()


def video_vote(preds):
    na = MMFIT_CLASSES.index("non_activity")
    active = [int(p) for p in preds if p != na]
    pool = active or [int(p) for p in preds]
    return Counter(pool).most_common(1)[0][0] if pool else na


def load_m2_rf():
    path = MODELS / "m2_rf.pickle"
    if not path.is_file():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def m2_p_correct(rf_pack, exercise: str, pose_norm: np.ndarray, start: int, end: int) -> float:
    if rf_pack is None:
        return float("nan")
    ex_key = {"squats": "squat", "lunges": "lunge", "plank": "plank"}.get(exercise)
    if ex_key is None or ex_key not in rf_pack.get("forests", {}):
        return float("nan")
    seg = pose_norm[start:end + 1]
    if len(seg) < 8 or not np.isfinite(seg).all():
        return float("nan")
    try:
        rep64 = resample(seg, 64)
        feats = feature_vector(features_2d(rep64)).reshape(1, -1)
        forest = rf_pack["forests"][ex_key]
        proba = forest.predict_proba(feats)[0]
        classes = list(forest.classes_)
        return float(proba[classes.index(0)]) if 0 in classes else float(proba.max())
    except Exception:  # noqa: BLE001
        return float("nan")


def evaluate(rows: list[dict]) -> dict:
    models = {name: load_model(name) for name in CHECKPOINTS if (MODELS / f"{name}.pt").exists()}
    rf_pack = load_m2_rf()
    primary = "m1_transformer_pose" if "m1_transformer_pose" in models else next(iter(models), None)

    set_rows = []
    window_y, window_p = {n: [] for n in models}, {n: [] for n in models}
    rep_true_ex, rep_pred_ex = [], []
    m2_probs = []
    skipped = []

    for row in rows:
        key = f"{row['person']}/{row['session_folder']}/set{int(row['set']):02d}"
        try:
            pose, df = load_set_pose(row)
        except FileNotFoundError:
            skipped.append((key, "no parquet"))
            continue
        start, end = activity_span(pose)
        segment = pose[start:end]
        detected = float(np.isfinite(segment).all(axis=(1, 2)).mean()) if len(segment) else 0.0
        x = windows(pose, start, end)
        y_idx = MMFIT_CLASSES.index(row["exercise"])
        votes = {}
        win_preds = {}
        for name, model in models.items():
            if len(x) == 0:
                votes[name] = MMFIT_CLASSES.index("non_activity")
                win_preds[name] = []
                continue
            p = predict(model, x)
            win_preds[name] = [MMFIT_CLASSES[int(i)] for i in p]
            window_y[name] += [y_idx] * len(p)
            window_p[name] += p.tolist()
            votes[name] = video_vote(p)

        pred_name = MMFIT_CLASSES[votes[primary]] if primary else None
        # Rep counting: oracle exercise + M1 predicted
        finite = segment[np.isfinite(segment).all(axis=(1, 2))]
        n_true = row["reps"]
        reps_source = row.get("reps_source") or "unknown"
        n_oracle = n_m1 = None
        if len(finite) >= 16:
            n_oracle, _ = count_reps(normalize(finite), exercise=row["exercise"])
            n_m1, _ = count_reps(normalize(finite), exercise=pred_name)
            if n_true is not None:
                entry = {
                    "key": key, "person": row["person"], "exercise": row["exercise"],
                    "true": int(n_true), "pred": int(n_oracle), "mode": "true_ex",
                    "reps_source": reps_source,
                }
                rep_true_ex.append(entry)
                rep_pred_ex.append({
                    "key": key, "person": row["person"], "exercise": row["exercise"],
                    "true": int(n_true), "pred": int(n_m1), "mode": "m1_ex",
                    "pred_exercise": pred_name, "reps_source": reps_source,
                })

        # Model 2 on squat/lunge/plank reps
        m2_set = []
        if row["exercise"] in ("squats", "lunges", "plank") and len(finite) >= 16:
            pose_n = normalize(finite)
            bounds = rep_boundaries(pose_n, exercise=row["exercise"])
            for a, b in bounds:
                p_corr = m2_p_correct(rf_pack, row["exercise"], pose_n, a, b)
                if np.isfinite(p_corr):
                    m2_set.append(p_corr)
                    m2_probs.append({
                        "key": key, "person": row["person"], "exercise": row["exercise"],
                        "use": row["use"], "p_correct": p_corr,
                        "pred_correct": int(p_corr >= 0.5),
                    })

        set_rows.append({
            "key": key,
            "person": row["person"],
            "session_folder": row["session_folder"],
            "set": int(row["set"]),
            "exercise": row["exercise"],
            "exercise_button": row.get("exercise_button") or "",
            "note": row.get("note") or "",
            "label_source": row.get("label_source"),
            "use": row["use"],
            "reps_true": n_true,
            "reps_oracle": n_oracle,
            "reps_m1": n_m1,
            "detected": detected,
            "n_windows": len(x),
            "votes": {n: MMFIT_CLASSES[v] for n, v in votes.items()},
            "vote_primary": pred_name,
            "correct_primary": bool(pred_name == row["exercise"]) if pred_name else False,
            "m2_p_correct_mean": float(np.mean(m2_set)) if m2_set else None,
            "hr_coverage": row.get("hr_coverage"),
            "hr_ok": row.get("hr_ok"),
            "hr_mean": row.get("hr_mean"),
            "hr_start": row.get("hr_start"),
            "hr_end": row.get("hr_end"),
        })

    eval_sets = [r for r in set_rows if r["use"] == "eval"]
    # M1 metrics on primary
    m1_out = {}
    for name in models:
        vy = np.array([MMFIT_CLASSES.index(r["exercise"]) for r in eval_sets])
        vp = np.array([MMFIT_CLASSES.index(r["votes"][name]) for r in eval_sets])
        k = int((vy == vp).sum())
        n = len(vy)
        lo, hi = wilson_ci(k, n)
        present = sorted({r["exercise"] for r in eval_sets})
        present_idx = [MMFIT_CLASSES.index(e) for e in present]
        # macro-F1 over the exercises present (exclude absent classes / non_activity)
        if n:
            from sklearn.metrics import f1_score
            f1 = float(f1_score(vy, vp, labels=present_idx, average="macro", zero_division=0))
        else:
            f1 = float("nan")
        per_person = {}
        for person in ("yassine", "partner"):
            m = np.array([r["person"] == person for r in eval_sets])
            if m.any():
                kk = int((vy[m] == vp[m]).sum())
                nn = int(m.sum())
                plo, phi = wilson_ci(kk, nn)
                per_person[person] = {"n": nn, "accuracy": kk / nn, "ci95": [plo, phi]}
        per_ex = {}
        for ex in present:
            m = vy == MMFIT_CLASSES.index(ex)
            nn = int(m.sum())
            kk = int((vy[m] == vp[m]).sum()) if nn else 0
            per_ex[ex] = {"n": nn, "accuracy": (kk / nn) if nn else None}
        wy = np.array(window_y[name])
        wp = np.array(window_p[name])
        # only windows from eval sets — approximate by filtering set_rows
        m1_out[name] = {
            "n_sets": n,
            "set_accuracy": (k / n) if n else None,
            "set_correct": k,
            "wilson95": [lo, hi],
            "macro_f1": float(f1) if n else None,
            "per_person": per_person,
            "per_exercise": per_ex,
            "window_accuracy": float((wy == wp).mean()) if len(wy) else None,
            "n_windows": int(len(wy)),
            "confusion": np.array(
                [[int(((vy == a) & (vp == b)).sum()) for b in range(len(MMFIT_CLASSES))]
                 for a in range(len(MMFIT_CLASSES))]
            ).tolist(),
        }

    def _rep_summary(items):
        if not items:
            return None
        t = [r["true"] for r in items]
        p = [r["pred"] for r in items]
        base = rep_count_errors(t, p)
        by_ex, by_person = {}, {}
        for r in items:
            by_ex.setdefault(r["exercise"], ([], []))
            by_ex[r["exercise"]][0].append(r["true"])
            by_ex[r["exercise"]][1].append(r["pred"])
            by_person.setdefault(r["person"], ([], []))
            by_person[r["person"]][0].append(r["true"])
            by_person[r["person"]][1].append(r["pred"])
        return {
            **base,
            "per_exercise": {e: rep_count_errors(a, b) for e, (a, b) in sorted(by_ex.items())},
            "per_person": {e: rep_count_errors(a, b) for e, (a, b) in sorted(by_person.items())},
            "rows": items,
        }

    # Main rep metrics: verified only. Typed defaults kept separately (provisional).
    def _eval_reps(items, source=None):
        out = []
        for r in items:
            s = next((x for x in set_rows if x["key"] == r["key"] and x["use"] == "eval"), None)
            if not s:
                continue
            if source is not None and r.get("reps_source") != source:
                continue
            out.append(r)
        return out

    has_verified_file = OWN_REPS_VERIFIED.is_file()
    if not has_verified_file:
        print("!" * 72)
        print("WARNING: data/own_reps_verified.csv missing.")
        print("Run:  python scripts/review_reps.py")
        print("Rep MAE numbers are PROVISIONAL (typed station defaults, not counts).")
        print("!" * 72)

    rep_oracle = _eval_reps(rep_true_ex, source="verified")
    rep_m1 = _eval_reps(rep_pred_ex, source="verified")
    rep_oracle_typed = _eval_reps(rep_true_ex, source="typed_default")
    rep_m1_typed = _eval_reps(rep_pred_ex, source="typed_default")

    m2_eval = [r for r in m2_probs if r["use"] == "eval" and r["exercise"] == "squats"]
    m2_all_squats = [r for r in m2_probs if r["exercise"] == "squats"]

    # Wrong-button check
    wrong_button = []
    for r in eval_sets:
        btn = r["exercise_button"]
        if btn and btn != r["exercise"]:
            wrong_button.append({
                "key": r["key"],
                "button": btn,
                "truth": r["exercise"],
                "m1": r["vote_primary"],
                "agrees_with": (
                    "note/truth" if r["vote_primary"] == r["exercise"]
                    else "button" if r["vote_primary"] == btn
                    else "neither"
                ),
            })

    # HR summary per person
    hr_summary = {}
    for person in ("yassine", "partner"):
        own = [r for r in rows if r["person"] == person]
        cov = [float(r["hr_coverage"]) for r in own if r.get("hr_coverage") is not None]
        ok = sum(1 for r in own if r.get("hr_ok"))
        rises = []
        for r in own:
            if r.get("hr_start") is not None and r.get("hr_end") is not None:
                if np.isfinite(r["hr_start"]) and np.isfinite(r["hr_end"]):
                    rises.append(float(r["hr_end"] - r["hr_start"]))
        hr_summary[person] = {
            "n_sets": len(own),
            "hr_ok_sets": ok,
            "mean_coverage": float(np.mean(cov)) if cov else 0.0,
            "median_coverage": float(np.median(cov)) if cov else 0.0,
            "mean_hr_rise": float(np.mean(rises)) if rises else None,
            "n_rise": len(rises),
        }

    return {
        "primary_model": primary,
        "model1": m1_out,
        "sets": set_rows,
        "rep_counting": {
            "true_exercise": _rep_summary(rep_oracle),
            "m1_exercise": _rep_summary(rep_m1),
            "typed_default_provisional": {
                "note": "typed station default (usually 10), NOT a counted ground truth",
                "true_exercise": _rep_summary(rep_oracle_typed),
                "m1_exercise": _rep_summary(rep_m1_typed),
            },
            "has_verified_file": has_verified_file,
            "n_verified_eval": len(rep_oracle),
            "n_typed_eval": len(rep_oracle_typed),
        },
        "m2": {
            "eval_squats": {
                "n_reps": len(m2_eval),
                "mean_p_correct": float(np.mean([r["p_correct"] for r in m2_eval])) if m2_eval else None,
                "pct_classified_correct": (
                    float(np.mean([r["pred_correct"] for r in m2_eval])) if m2_eval else None
                ),
            },
            "all_squats_incl_test": {
                "n_reps": len(m2_all_squats),
                "mean_p_correct": float(np.mean([r["p_correct"] for r in m2_all_squats])) if m2_all_squats else None,
            },
            "rows": m2_probs,
        },
        "wrong_button": wrong_button,
        "hr_summary": hr_summary,
        "skipped": skipped,
        "n_eval": len(eval_sets),
        "n_test": sum(1 for r in set_rows if r["use"] == "test"),
    }


def plot_hr_session(person: str, session_folder: str, out: Path):
    session_dir = OWN_DIR / person / session_folder
    hr_path = session_dir / "hr.csv"
    if not hr_path.is_file():
        return None
    hr = pd.read_csv(hr_path)
    sets = pd.read_csv(session_dir / "sets.csv") if (session_dir / "sets.csv").is_file() else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    t0 = float(hr["pc_ts_ms"].iloc[0])
    ax.plot((hr["pc_ts_ms"] - t0) / 1000.0, hr["bpm"], color="#C23B22", lw=1.2, label="HR")
    for _, s in sets.iterrows():
        a = (float(s["start_ms"]) - t0) / 1000.0
        b = (float(s["end_ms"]) - t0) / 1000.0
        ax.axvspan(a, b, color="#1E4BD2", alpha=0.15)
        ax.text((a + b) / 2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 140,
                f"set {int(s['set'])}", ha="center", fontsize=8, color="#1E4BD2")
    ax.set_xlabel("seconds from first HR sample")
    ax.set_ylabel("bpm")
    ax.set_title(f"HR · {person}/{session_folder}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def write_report(result: dict, run_dir: Path):
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    primary = result["primary_model"]
    m1 = result["model1"].get(primary, {})
    rc_t = result["rep_counting"]["true_exercise"]
    rc_m = result["rep_counting"]["m1_exercise"]
    m2 = result["m2"]["eval_squats"]

    # Confusion figure
    if m1.get("confusion"):
        cm = np.array(m1["confusion"])
        plot_confusion(cm, MMFIT_CLASSES, FIGURES / "m1_confusion_own.png",
                       title=f"{primary} on own videos (set vote)")
        plot_confusion(cm, MMFIT_CLASSES, run_dir / "m1_confusion_own.png",
                       title=f"{primary} on own videos (set vote)")

    # Rep figure
    if rc_t and rc_t.get("rows"):
        fig, ax = plt.subplots(figsize=(6, 5))
        truths = [r["true"] for r in rc_t["rows"]]
        preds = [r["pred"] for r in rc_t["rows"]]
        ax.scatter(truths, preds, c="#1E4BD2", alpha=0.8)
        lim = max(max(truths), max(preds)) + 1
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlabel("true reps")
        ax.set_ylabel("counted reps (true exercise)")
        ax.set_title(f"Own videos rep counting (n={rc_t['n']}, MAE={rc_t['mae']:.2f})")
        fig.tight_layout()
        fig.savefig(FIGURES / "rep_counting_own.png", dpi=120)
        fig.savefig(run_dir / "rep_counting_own.png", dpi=120)
        plt.close(fig)

    # HR figure: prefer TRICEPS or partner s01
    for person, folder in (("yassine", "2026-11-10_TRICEPS"), ("partner", "2026-11-10_s01")):
        path = plot_hr_session(person, folder, FIGURES / "own_hr_session.png")
        if path:
            plot_hr_session(person, folder, run_dir / "own_hr_session.png")
            break

    lo, hi = m1.get("wilson95", [None, None])
    lines = [
        "# RQ4 · Models 1–2 on our own recordings",
        "",
        "Gym station (iPhone portrait 720×1280 @ 30 fps + Galaxy Watch 7 HR), two people, one day.",
        "Labels from `data/own_manifest.csv` (note wins over the exercise button). **Test data only — no training or threshold tuning.**",
        "",
        f"Eval sets: **{result['n_eval']}** (partner 8 + yassine 12). Test sets (excluded from main numbers): **{result['n_test']}** "
        "(yassine 2026-11-09_s01 lunges; yassine 2026-11-10_s01 squats, note=`tt`).",
        "Exercises covered in eval: 9 of 10 MM-Fit (no lunges except the test set). All form labels = correct.",
        "RPE ignored (almost every set typed 10). Weight typed as body weight — ignored.",
        "",
        "## Label corrections",
        "",
        "Wrong button pressed on 3 eval sets; the note (and folder name for 2 unsaved sets) is the truth:",
        "",
        "| Set | Button | Note / folder | Truth |",
        "|---|---|---|---|",
    ]
    for r in result["sets"]:
        if r["use"] != "eval":
            continue
        if (r["exercise_button"] and r["exercise_button"] != r["exercise"]) or r["label_source"] == "folder_name":
            lines.append(
                f"| `{r['key']}` | {r['exercise_button'] or '—'} | {r['note'] or r['session_folder']} | **{r['exercise']}** |"
            )
    lines += [
        "",
        "## Model 1 (pose only)",
        "",
        f"Primary checkpoint: `{primary}`. Windows 90 / hop 45, majority vote excluding `non_activity`.",
        f"Set accuracy: **{m1.get('set_accuracy')}** ({m1.get('set_correct')}/{m1.get('n_sets')}), "
        f"Wilson 95% CI [{lo:.2f}, {hi:.2f}]. Macro-F1 (9 exercises present): **{m1.get('macro_f1')}**.",
        f"Window accuracy: **{m1.get('window_accuracy')}** (n={m1.get('n_windows')}).",
        "",
        "Differences of a few sets are noise at this n.",
        "",
        "| Person | n | Accuracy | Wilson 95% |",
        "|---|---:|---:|---|",
    ]
    for person, v in (m1.get("per_person") or {}).items():
        lines.append(f"| {person} | {v['n']} | {v['accuracy']:.3f} | [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] |")
    lines += ["", "| Exercise | n | Accuracy |", "|---|---:|---:|"]
    for ex, v in (m1.get("per_exercise") or {}).items():
        acc = "–" if v["accuracy"] is None else f"{v['accuracy']:.3f}"
        lines.append(f"| {ex} | {v['n']} | {acc} |")

    lines += [
        "",
        "### Wrong-button check",
        "",
        "Did Model 1 agree with the note (truth) or the wrong button?",
        "",
        "| Set | Button | Truth | M1 | Agrees with |",
        "|---|---|---|---|---|",
    ]
    for w in result["wrong_button"]:
        lines.append(f"| `{w['key']}` | {w['button']} | {w['truth']} | {w['m1']} | **{w['agrees_with']}** |")

    lines += [
        "",
        "### Comparison (same pose-only transformer, set/video accuracy)",
        "",
        "| Source | Metric | Value | n |",
        "|---|---|---:|---:|",
        "| MM-Fit test_unseen (dataset pose) | macro-F1 | 0.85 | — |",
        "| MM-Fit w20 MediaPipe | macro-F1 | 0.80 | — |",
        "| RepCount public videos | video accuracy (4 ex.) | 0.808 | — |",
        f"| **Our videos** | set accuracy | {m1.get('set_accuracy')} | {m1.get('n_sets')} |",
        f"| **Our videos** | macro-F1 (9 ex.) | {m1.get('macro_f1')} | {m1.get('n_sets')} |",
        "",
        "Confusion: `reports/figures/m1_confusion_own.png`.",
        "",
    ]

    lines += [
        "## Rep counting",
        "",
        "No peak-parameter tuning on these sets. **Main numbers use hand-verified reps only** "
        f"(`reps_source=verified`).",
        "",
    ]
    rc_block = result["rep_counting"]
    if not rc_block.get("has_verified_file"):
        lines += [
            "> **PROVISIONAL:** `data/own_reps_verified.csv` is missing. "
            "Run `python scripts/review_reps.py` and count the 22 sets. "
            "Do not cite typed-default MAE as a result.",
            "",
        ]
    if rc_t and rc_t.get("n"):
        lines += [
            f"Verified · true exercise: MAE **{rc_t['mae']:.2f}**, exact **{rc_t['exact']*100:.1f}%**, "
            f"within ±1 **{rc_t['within_1']*100:.1f}%** (n={rc_t['n']}).",
        ]
    elif rc_block.get("has_verified_file"):
        lines += ["Verified file present but **0** verified eval sets with counts yet.", ""]
    if rc_m and rc_m.get("n"):
        lines += [
            f"Verified · M1-predicted exercise: MAE **{rc_m['mae']:.2f}**, exact **{rc_m['exact']*100:.1f}%**, "
            f"within ±1 **{rc_m['within_1']*100:.1f}%** (n={rc_m['n']}).",
            "",
            "| Exercise | n | MAE (true ex) |",
            "|---|---:|---:|",
        ]
        for ex, v in (rc_t.get("per_exercise") or {}).items():
            lines.append(f"| {ex} | {v['n']} | {v['mae']:.2f} |")
        lines += ["", "| Person | n | MAE (true ex) |", "|---|---:|---:|"]
        for person, v in (rc_t.get("per_person") or {}).items():
            lines.append(f"| {person} | {v['n']} | {v['mae']:.2f} |")
    typed = (rc_block.get("typed_default_provisional") or {}).get("true_exercise")
    if typed and typed.get("n"):
        lines += [
            "",
            f"**Typed default (not a count):** MAE {typed['mae']:.2f}, exact {typed['exact']*100:.1f}%, "
            f"±1 {typed['within_1']*100:.1f}% (n={typed['n']}). "
            "Station UI used to default reps=10 — these numbers are **not** ground truth.",
        ]
    lines += [
        "",
        "Figure: `reports/figures/rep_counting_own.png`.",
        "",
        "## Model 2 (form)",
        "",
        "Only **correct** squats recorded (2 eval + 1 test). No lunges/plank mistakes. "
        "This measures **false alarms**, not mistake detection.",
        f"Eval squats: mean P(correct) **{m2.get('mean_p_correct')}**, "
        f"% reps classified correct **{m2.get('pct_classified_correct')}** (n={m2.get('n_reps')}).",
        "Compare MM-Fit RQ4 MediaPipe P(correct) on squat/lunge: **0.08 / 0.14**.",
        "",
        "## Heart rate",
        "",
        "| Person | sets | hr_ok | mean coverage | median coverage | mean HR rise (end−start) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for person, h in result["hr_summary"].items():
        rise = "–" if h["mean_hr_rise"] is None else f"{h['mean_hr_rise']:.1f}"
        lines.append(
            f"| {person} | {h['n_sets']} | {h['hr_ok_sets']} | {h['mean_coverage']:.2f} | "
            f"{h['median_coverage']:.2f} | {rise} |"
        )
    lines += [
        "",
        "Limitation: the watch went silent for 20–80 s on several sets (hr_coverage 0 on many early sets).",
        "Figure: `reports/figures/own_hr_session.png`.",
        "",
        "## Limits",
        "",
        "- Small n (20 eval sets), one day, two people, 9 exercises, no lunges in eval.",
        "- Portrait phone framing and gym background differ from MM-Fit / RepCount.",
        "- RPE not usable; HR gaps; two sets missing typed reps.",
        "- Own data never used for training or threshold tuning in this report.",
        "",
        f"Metrics: `{run_dir.as_posix()}/metrics.json`.",
        "",
    ]
    text = "\n".join(lines)
    (REPORTS / "rq4_own.md").write_text(text, encoding="utf-8")
    (run_dir / "rq4_own.md").write_text(text, encoding="utf-8")

    # Update rq4_first_look with a short pointer
    first = REPORTS / "rq4_first_look.md"
    if first.is_file():
        body = first.read_text(encoding="utf-8")
        marker = "## Own recordings (Nov 10, 2026)"
        if marker not in body:
            acc = m1.get("set_accuracy")
            f1 = m1.get("macro_f1")
            mae = rc_t["mae"] if rc_t else None
            blurb = (
                f"\n{marker}\n\n"
                f"Real gym-station videos (2 people, 20 eval sets). "
                f"Pose-only transformer set accuracy **{acc}** (Wilson CI "
                f"[{lo:.2f}, {hi:.2f}]), macro-F1 **{f1}**, rep MAE **{mae}**. "
                f"Full tables: [rq4_own.md](rq4_own.md).\n"
            )
            first.write_text(body.rstrip() + "\n" + blurb, encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--session", type=str, default=None, help="person/session_folder filter")
    p.add_argument("--manifest", type=str, default=None)
    args = p.parse_args(argv)

    rows = load_manifest(Path(args.manifest) if args.manifest else OWN_MANIFEST)
    if args.session:
        part = args.session.replace("\\", "/").strip("/")
        if "/" in part:
            person, folder = part.split("/", 1)
            rows = [r for r in rows if r["person"] == person and r["session_folder"] == folder]
        else:
            rows = [r for r in rows if r["session_folder"] == part]
    rows = [r for r in rows if r["use"] in ("eval", "test")]

    result = evaluate(rows)
    run_dir = new_run("own_rq4", {
        "n_eval": result["n_eval"],
        "n_test": result["n_test"],
        "primary": result["primary_model"],
        "checkpoints": list(result["model1"]),
        "seed": 42,
    })
    # JSON-safe metrics (drop huge row dumps partially)
    metrics = {
        "primary_model": result["primary_model"],
        "model1": result["model1"],
        "rep_counting": {
            "true_exercise": {k: v for k, v in (result["rep_counting"]["true_exercise"] or {}).items() if k != "rows"},
            "m1_exercise": {k: v for k, v in (result["rep_counting"]["m1_exercise"] or {}).items() if k != "rows"},
            "typed_default_provisional": {
                "true_exercise": {
                    k: v for k, v in (
                        (result["rep_counting"].get("typed_default_provisional") or {})
                        .get("true_exercise") or {}
                    ).items() if k != "rows"
                },
            },
            "has_verified_file": result["rep_counting"].get("has_verified_file"),
            "n_verified_eval": result["rep_counting"].get("n_verified_eval"),
            "n_typed_eval": result["rep_counting"].get("n_typed_eval"),
        },
        "m2": {k: v for k, v in result["m2"].items() if k != "rows"},
        "wrong_button": result["wrong_button"],
        "hr_summary": result["hr_summary"],
        "sets": result["sets"],
        "skipped": result["skipped"],
        "n_eval": result["n_eval"],
        "n_test": result["n_test"],
    }
    save_metrics(run_dir, metrics)
    write_report(result, run_dir)
    m1 = result["model1"].get(result["primary_model"], {})
    print(f"M1 set acc {m1.get('set_accuracy')} CI {m1.get('wilson95')} macro-F1 {m1.get('macro_f1')}")
    rc = result["rep_counting"]["true_exercise"]
    if rc and rc.get("n"):
        print(f"rep MAE (verified) {rc['mae']:.2f} exact {rc['exact']:.2f} ±1 {rc['within_1']:.2f} n={rc['n']}")
    else:
        print("rep MAE (verified): n=0 — run python scripts/review_reps.py")
    typed = (result["rep_counting"].get("typed_default_provisional") or {}).get("true_exercise")
    if typed and typed.get("n"):
        print(
            f"rep MAE (typed default, not a count) {typed['mae']:.2f} "
            f"exact {typed['exact']:.2f} ±1 {typed['within_1']:.2f} n={typed['n']}"
        )
    print(f"M2 P(correct) {result['m2']['eval_squats']}")
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
