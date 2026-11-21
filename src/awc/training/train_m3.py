"""Train / evaluate Model 3 variants B1–B5 on MM-Fit (no fine-tune stage).

Fair B1 uses a continuous speed-loss score for AUC (not a 0/1 step).
Alert thresholds for B2–B5 are chosen on val (Youden or ≤1 FA/session).
Never tune on test_unseen.

    python -m awc.training.train_m3 --all
    python -m awc.training.train_m3 --all --quick
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder

from awc.config import MMFIT_CLASSES, PROCESSED, models_dir
from awc.features.fatigue_dataset import FEATURE_COLS, M3_DIR, build_all
from awc.models.m3_fatigue import M3Model, pad_batch, rule_baseline_alert
from awc.training.tracking import new_run, save_metrics, seed_everything

IN_DIM = len(FEATURE_COLS) + 2
SEEDS = (42, 43, 44)


def load_npz(name="general"):
    path = M3_DIR / f"sets_{name}.npz"
    if not path.is_file():
        build_all(use_form=False)
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _records(pack, split):
    idx = [i for i, s in enumerate(pack["split"]) if s == split]
    out = []
    for i in idx:
        x = pack["X"][i]
        m = pack["masks"][i]
        xm = np.concatenate([x, m], axis=1).astype(np.float32)
        raw_speed = pack["raw_speed"][i] if "raw_speed" in pack else pack["X"][i][:, FEATURE_COLS.index("mean_speed")]
        out.append({
            "X": xm,
            "fatigue": pack["fatigue"][i].astype(np.float32),
            "risk": pack["risk"][i].astype(np.float32),
            "session": str(pack["session"][i]),
            "exercise": str(pack["exercise"][i]),
            "set_id": int(pack["set_id"][i]),
            "set_idx": int(pack["set_idx_in_exercise"][i]),
            "breakdown_i": int(pack["breakdown_i"][i]),
            "speed": np.asarray(raw_speed, dtype=np.float64),
            "n": int(len(xm)),
            "hr_mask": pack["masks"][i][:, 1].astype(np.float32),
        })
    return out


def _apply_hr(records, hr_on: bool):
    out = []
    for r in records:
        x = r["X"].copy()
        if not hr_on:
            hr_end_i = FEATURE_COLS.index("hr_end")
            hr_rise_i = FEATURE_COLS.index("hr_rise")
            x[:, hr_end_i] = 0.0
            x[:, hr_rise_i] = 0.0
            x[:, len(FEATURE_COLS) + 1] = 0.0
        nr = dict(r)
        nr["X"] = x
        out.append(nr)
    return out


def _batches(records, batch=32, shuffle=False):
    order = np.arange(len(records))
    if shuffle:
        np.random.shuffle(order)
    for start in range(0, len(order), batch):
        sl = [records[i] for i in order[start:start + batch]]
        x, lengths, _ = pad_batch([s["X"] for s in sl])
        fat, _, _ = pad_batch([s["fatigue"][:, None] for s in sl])
        risk, _, _ = pad_batch([s["risk"][:, None] for s in sl])
        yield x, lengths, fat.squeeze(-1), risk.squeeze(-1), sl


def train_model(records_train, records_val, epochs=30, lr=1e-3, batch=32, quick=False, pos_weight=3.0, seed=42):
    seed_everything(seed)
    if quick:
        epochs = 1
        records_train = records_train[:min(64, len(records_train))]
        records_val = records_val[:min(32, len(records_val))]
    model = M3Model(in_dim=IN_DIM, hidden=32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss(reduction="none")
    mse = nn.MSELoss(reduction="none")
    best_state, best_auc = None, -1.0
    for _epoch in range(epochs):
        model.train()
        for x, lengths, fat, risk, _sl in _batches(records_train, batch=batch, shuffle=True):
            pred_f, pred_r = model(x, lengths)
            mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]
            loss_f = (mse(pred_f, fat) * mask).sum() / mask.sum().clamp_min(1)
            w = torch.where(risk > 0.5, torch.full_like(risk, pos_weight), torch.ones_like(risk))
            loss_r = (bce(pred_r, risk) * w * mask).sum() / mask.sum().clamp_min(1)
            loss = loss_f + loss_r
            opt.zero_grad()
            loss.backward()
            opt.step()
        metrics = evaluate_scores(*_predict(model, records_val), records_val, thresh=0.7)
        if metrics["risk_auc"] is not None and metrics["risk_auc"] >= best_auc:
            best_auc = metrics["risk_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict(model, records):
    model.eval()
    fats, risks = [], []
    with torch.no_grad():
        for x, lengths, _f, _r, sl in _batches(records, batch=64, shuffle=False):
            pf, pr = model(x, lengths)
            for i, s in enumerate(sl):
                n = len(s["X"])
                fats.append(pf[i, :n].cpu().numpy())
                risks.append(pr[i, :n].cpu().numpy())
    return fats, risks


def speed_loss_scores(speeds):
    """Continuous score per rep: max(0, 1 - speed / mean(first 2))."""
    speeds = np.asarray(speeds, dtype=np.float64)
    n = len(speeds)
    out = np.zeros(n, dtype=np.float64)
    if n < 2:
        return out
    base = np.nanmean(speeds[:2])
    if not np.isfinite(base) or base <= 0:
        return out
    for i in range(n):
        if np.isfinite(speeds[i]):
            out[i] = max(0.0, 1.0 - float(speeds[i]) / base)
    return out


def warning_lead_time(risk_pred, breakdown_i, thresh=0.7):
    if breakdown_i is None or breakdown_i < 0:
        return None
    alerts = np.where(np.asarray(risk_pred) >= thresh)[0]
    if len(alerts) == 0:
        return 0.0
    first = int(alerts[0])
    if first > breakdown_i:
        return 0.0
    return float(breakdown_i - first)


def false_alarms_per_session(records, risk_preds, thresh=0.7):
    by_sess = {}
    for r, pred in zip(records, risk_preds):
        bd = r["breakdown_i"]
        if bd is not None and bd >= 0:
            continue
        alerts = int(np.sum(np.asarray(pred) >= thresh))
        by_sess.setdefault(r["session"], 0)
        by_sess[r["session"]] += alerts
    if not by_sess:
        return 0.0
    return float(np.mean(list(by_sess.values())))


def _set_speed_loss(r):
    """Measured end-of-set speed loss vs first-2 mean (for Spearman)."""
    sp = r["speed"]
    if len(sp) < 3:
        return np.nan
    base = np.nanmean(sp[:2])
    if not np.isfinite(base) or base <= 0:
        return np.nan
    return float(1.0 - sp[-1] / base)


def evaluate_scores(fats, risks, records, thresh=0.7):
    if not records:
        return {
            "risk_auc": None, "lead_time": None, "pct_warned_1plus": None,
            "false_alarms": None, "spearman_set_idx": None, "spearman_speed_loss": None,
            "n_sets": 0, "thresh": thresh,
        }
    y_true = np.concatenate([r["risk"] for r in records])
    y_pred = np.concatenate(risks)
    auc = float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None
    leads, warned = [], []
    for r, pred in zip(records, risks):
        bd = r["breakdown_i"] if r["breakdown_i"] >= 0 else None
        lt = warning_lead_time(pred, bd, thresh=thresh)
        if lt is not None:
            leads.append(lt)
            warned.append(1.0 if lt >= 1.0 else 0.0)
    last_fat = [float(f[-1]) for f in fats]
    set_idx = [r["set_idx"] for r in records]
    speed_losses = [_set_speed_loss(r) for r in records]
    spearman_idx = None
    if len(set_idx) >= 3 and len(set(set_idx)) > 1:
        spearman_idx = float(spearmanr(last_fat, set_idx).correlation)
    spearman_sl = None
    ok = np.isfinite(speed_losses)
    if ok.sum() >= 3:
        spearman_sl = float(spearmanr(np.asarray(last_fat)[ok], np.asarray(speed_losses)[ok]).correlation)
    return {
        "risk_auc": auc,
        "lead_time": float(np.mean(leads)) if leads else None,
        "pct_warned_1plus": float(np.mean(warned)) if warned else None,
        "false_alarms": false_alarms_per_session(records, risks, thresh=thresh),
        "spearman_set_idx": spearman_idx,
        "spearman_speed_loss": spearman_sl,
        "n_sets": len(records),
        "thresh": float(thresh),
    }


def evaluate_model(model, records, thresh=0.7):
    fats, risks = _predict(model, records)
    return evaluate_scores(fats, risks, records, thresh=thresh)


def pick_threshold_on_val(risks, records, mode="youden"):
    """Choose alert threshold on val only. mode: youden | fa1."""
    y_true = np.concatenate([r["risk"] for r in records])
    y_score = np.concatenate(risks)
    if len(np.unique(y_true)) < 2:
        return 0.7
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    if mode == "youden":
        j = tpr - fpr
        return float(thresholds[int(np.argmax(j))])
    # ≤ 1 false alarm per session: try descending thresholds
    cands = sorted({float(t) for t in thresholds if np.isfinite(t)}, reverse=True)
    cands = [t for t in cands if 0.05 <= t <= 0.95] or [0.7]
    best = cands[0]
    for t in cands:
        fa = false_alarms_per_session(records, risks, thresh=t)
        if fa <= 1.0:
            best = t
            break
    return best


def evaluate_rule(records, drop=0.25, alert_drop=None):
    """B1 with continuous speed-loss score for AUC; binary alert at alert_drop (default=drop)."""
    alert_drop = drop if alert_drop is None else alert_drop
    scores, binary = [], []
    for r in records:
        sc = speed_loss_scores(r["speed"])
        scores.append(sc.astype(np.float32))
        pred = (sc >= alert_drop).astype(np.float32)
        # once alerted, stay on (match "stop" through end of set)
        if pred.any():
            first = int(np.argmax(pred))
            pred[first:] = 1.0
        binary.append(pred)
    # AUC on continuous scores vs risk labels
    yt = np.concatenate([r["risk"] for r in records])
    ys = np.concatenate(scores)
    auc = float(roc_auc_score(yt, ys)) if len(np.unique(yt)) > 1 else None
    # lead / FA use the binary alerts at alert_drop
    # fabricate "fats" as zeros for spearman slots
    fats = [np.zeros(len(r["X"]), dtype=np.float32) for r in records]
    metrics = evaluate_scores(fats, binary, records, thresh=0.5)
    metrics["risk_auc"] = auc  # override with continuous-score AUC
    metrics["drop"] = float(drop)
    alert_rate = float(np.mean([1.0 if p.any() else 0.0 for p in binary]))
    metrics["alert_rate"] = alert_rate
    return metrics


def evaluate_exercise_control(train, test):
    """Logistic regression on one-hot exercise + rep_idx/n only."""
    def rows(recs):
        xs, ys = [], []
        for r in recs:
            n = r["n"]
            for t in range(n):
                xs.append([r["exercise"], t / max(n - 1, 1)])
                ys.append(float(r["risk"][t]))
        return xs, np.asarray(ys)

    x_tr, y_tr = rows(train)
    x_te, y_te = rows(test)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ex_tr = enc.fit_transform([[row[0]] for row in x_tr])
    ex_te = enc.transform([[row[0]] for row in x_te])
    Xtr = np.column_stack([ex_tr, [row[1] for row in x_tr]])
    Xte = np.column_stack([ex_te, [row[1] for row in x_te]])
    if len(np.unique(y_tr)) < 2:
        return {"risk_auc": None, "n_sets": len(test)}
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(Xtr, y_tr)
    proba = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(y_te, proba)) if len(np.unique(y_te)) > 1 else None
    return {"risk_auc": auc, "n_sets": len(test), "name": "exercise_only"}


def spearman_diagnostics(records):
    """Explain negative Spearman(fatigue, set_idx): fewer reps later? exercise order?"""
    n_reps = [r["n"] for r in records]
    set_idx = [r["set_idx"] for r in records]
    corr_n = float(spearmanr(n_reps, set_idx).correlation) if len(set(set_idx)) > 1 else None
    # exercise order: first occurrence index in session
    return {
        "spearman_n_reps_vs_set_idx": corr_n,
        "mean_n_reps_by_set_idx": {
            str(k): float(np.mean([r["n"] for r in records if r["set_idx"] == k]))
            for k in sorted(set(set_idx))
        },
    }


VARIANTS = {
    "B1": {"norm": None, "hr": None, "rule": True},
    "B2": {"norm": "general", "hr": True, "rule": False},
    "B3": {"norm": "personal", "hr": False, "rule": False},
    "B4": {"norm": "personal", "hr": True, "rule": False},
    "B5": {"norm": "general", "hr": False, "rule": False},
}


def _mean_std(values):
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if len(arr) == 0:
        return None, None
    return float(arr.mean()), float(arr.std(ddof=0))


def run_variant(name, quick=False, epochs=30, seeds=SEEDS, thresh_mode="youden"):
    cfg = VARIANTS[name]
    run_name = f"m3_{name.lower()}_v2" + ("_quick" if quick else "")
    run = new_run(run_name, {"variant": name, **cfg, "quick": quick, "seeds": list(seeds)})
    if cfg["rule"]:
        pack = load_npz("general")
        test = _records(pack, "test_unseen")
        m25 = evaluate_rule(test, drop=0.25)
        m20 = evaluate_rule(test, drop=0.20)
        metrics = {
            "variant": "B1",
            "b1_drop_25": m25,
            "b1_drop_20": m20,
            "risk_auc": m25["risk_auc"],
            "lead_time": m25["lead_time"],
            "pct_warned_1plus": m25["pct_warned_1plus"],
            "false_alarms": m25["false_alarms"],
            "alert_rate_25": m25["alert_rate"],
            "alert_rate_20": m20["alert_rate"],
            "spearman_diagnostics": spearman_diagnostics(test),
        }
        save_metrics(run, metrics)
        return metrics

    pack = load_npz(cfg["norm"])
    train = _apply_hr(_records(pack, "train"), cfg["hr"])
    val = _apply_hr(_records(pack, "val"), cfg["hr"])
    test = _apply_hr(_records(pack, "test_unseen"), cfg["hr"])

    seed_metrics = []
    best_model = None
    chosen_thresh = 0.7
    for seed in (seeds if not quick else (42,)):
        model = train_model(train, val, epochs=epochs, quick=quick, seed=seed)
        _, val_risks = _predict(model, val)
        thresh = pick_threshold_on_val(val_risks, val, mode=thresh_mode)
        m_val_tuned = evaluate_model(model, test, thresh=thresh)
        m_07 = evaluate_model(model, test, thresh=0.7)
        seed_metrics.append({
            "seed": seed,
            "thresh_val": thresh,
            "val_tuned": m_val_tuned,
            "thresh_0.7": m_07,
        })
        best_model = model
        chosen_thresh = thresh

    # aggregate val-tuned
    keys = ["risk_auc", "lead_time", "pct_warned_1plus", "false_alarms", "spearman_set_idx", "spearman_speed_loss"]
    agg = {}
    for k in keys:
        mean, std = _mean_std([s["val_tuned"][k] for s in seed_metrics])
        agg[k] = mean
        agg[f"{k}_std"] = std
    agg_07 = {}
    for k in keys:
        mean, std = _mean_std([s["thresh_0.7"][k] for s in seed_metrics])
        agg_07[k] = mean
        agg_07[f"{k}_std"] = std

    control = evaluate_exercise_control(train, test)
    metrics = {
        "variant": name,
        "thresh_mode": thresh_mode,
        "thresh_val_mean": float(np.mean([s["thresh_val"] for s in seed_metrics])),
        "val_tuned": agg,
        "thresh_0.7": agg_07,
        "seeds": seed_metrics,
        "exercise_only_auc": control.get("risk_auc"),
        "spearman_diagnostics": spearman_diagnostics(test),
        "risk_auc": agg["risk_auc"],
        "lead_time": agg["lead_time"],
        "pct_warned_1plus": agg["pct_warned_1plus"],
        "false_alarms": agg["false_alarms"],
        "spearman_set_idx": agg["spearman_set_idx"],
        "spearman_speed_loss": agg["spearman_speed_loss"],
    }
    if name == "B4":
        w05 = [r for r in test if r["session"] == "w05"]
        if w05 and best_model is not None:
            metrics["w05_no_hr"] = evaluate_model(best_model, w05, thresh=chosen_thresh)
    if best_model is not None:
        out = models_dir()
        out.mkdir(parents=True, exist_ok=True)
        ckpt_name = f"m3_{name.lower()}" + ("_quick" if quick else "") + ".pt"
        torch.save({
            "variant": name, "state_dict": best_model.state_dict(),
            "in_dim": IN_DIM, "thresh": chosen_thresh,
        }, out / ckpt_name)
    save_metrics(run, metrics)
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANTS), default="B4")
    p.add_argument("--all", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--thresh-mode", choices=["youden", "fa1"], default="youden")
    args = p.parse_args(argv)
    if args.rebuild or not (M3_DIR / "sets_general.npz").is_file():
        print(build_all(use_form=False), flush=True)
    names = list(VARIANTS) if args.all else [args.variant]
    results = [
        run_variant(n, quick=args.quick, epochs=args.epochs, thresh_mode=args.thresh_mode)
        for n in names
    ]
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
