"""Train / evaluate Model 3 variants B1–B5 on MM-Fit (no fine-tune stage).

    python -m awc.training.train_m3 --variant B4
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
from sklearn.metrics import roc_auc_score

from awc.config import MODELS, PROCESSED
from awc.features.fatigue_dataset import FEATURE_COLS, M3_DIR, build_all
from awc.models.m3_fatigue import M3Model, pad_batch, rule_baseline_alert
from awc.training.tracking import new_run, save_metrics, seed_everything

# X channels: features + masks
IN_DIM = len(FEATURE_COLS) + 2


def load_npz(name="general"):
    path = M3_DIR / f"sets_{name}.npz"
    if not path.is_file():
        build_all()
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
            "hr_mask": pack["masks"][i][:, 1].astype(np.float32),
        })
    return out


def _apply_hr(records, hr_on: bool):
    out = []
    for r in records:
        x = r["X"].copy()
        if not hr_on:
            # zero HR feature channels + mask
            hr_end_i = FEATURE_COLS.index("hr_end")
            hr_rise_i = FEATURE_COLS.index("hr_rise")
            x[:, hr_end_i] = 0.0
            x[:, hr_rise_i] = 0.0
            x[:, len(FEATURE_COLS) + 1] = 0.0  # hr_feat_mask
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


def train_model(records_train, records_val, epochs=30, lr=1e-3, batch=32, quick=False, pos_weight=3.0):
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
            # mask padded steps
            mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]
            loss_f = (mse(pred_f, fat) * mask).sum() / mask.sum().clamp_min(1)
            w = torch.where(risk > 0.5, torch.full_like(risk, pos_weight), torch.ones_like(risk))
            loss_r = (bce(pred_r, risk) * w * mask).sum() / mask.sum().clamp_min(1)
            loss = loss_f + loss_r
            opt.zero_grad()
            loss.backward()
            opt.step()
        metrics = evaluate_model(model, records_val)
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


def warning_lead_time(risk_pred, breakdown_i, thresh=0.7):
    """Reps between first alert and breakdown; 0 if alert after / never / no breakdown."""
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


def evaluate_model(model, records, thresh=0.7):
    if not records:
        return {"risk_auc": None, "lead_time": None, "false_alarms": None, "spearman_set_idx": None}
    fats, risks = _predict(model, records)
    y_true = np.concatenate([r["risk"] for r in records])
    y_pred = np.concatenate(risks)
    auc = None
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_pred))
    leads = []
    for r, pred in zip(records, risks):
        lt = warning_lead_time(pred, r["breakdown_i"] if r["breakdown_i"] >= 0 else None, thresh=thresh)
        if lt is not None:
            leads.append(lt)
    # Spearman: last-rep predicted fatigue vs set_idx_in_exercise
    last_fat = [float(f[-1]) for f in fats]
    set_idx = [r["set_idx"] for r in records]
    spearman = None
    if len(set_idx) >= 3 and len(set(set_idx)) > 1:
        spearman = float(spearmanr(last_fat, set_idx).correlation)
    return {
        "risk_auc": auc,
        "lead_time": float(np.mean(leads)) if leads else None,
        "false_alarms": false_alarms_per_session(records, risks, thresh=thresh),
        "spearman_set_idx": spearman,
        "n_sets": len(records),
    }


def evaluate_rule(records, drop=0.25, thresh_as_alert=True):
    """B1 rule on raw mean_speed (stored beside the normalised features)."""
    leads, y_true, y_score = [], [], []
    preds = []
    for r in records:
        alert = rule_baseline_alert(r["speed"], drop=drop)
        pred = np.zeros(len(r["X"]), dtype=np.float32)
        if alert is not None:
            pred[alert:] = 1.0
        preds.append(pred)
        y_true.append(r["risk"])
        y_score.append(pred)
        lt = warning_lead_time(pred, r["breakdown_i"] if r["breakdown_i"] >= 0 else None, thresh=0.7)
        if lt is not None:
            leads.append(lt)
    yt = np.concatenate(y_true)
    ys = np.concatenate(y_score)
    auc = float(roc_auc_score(yt, ys)) if len(np.unique(yt)) > 1 else None
    return {
        "risk_auc": auc,
        "lead_time": float(np.mean(leads)) if leads else None,
        "false_alarms": false_alarms_per_session(records, preds, thresh=0.7),
        "spearman_set_idx": None,
        "n_sets": len(records),
    }


VARIANTS = {
    "B1": {"norm": None, "hr": None, "rule": True},
    "B2": {"norm": "general", "hr": True, "rule": False},
    "B3": {"norm": "personal", "hr": False, "rule": False},
    "B4": {"norm": "personal", "hr": True, "rule": False},
    "B5": {"norm": "general", "hr": False, "rule": False},
}


def run_variant(name, quick=False, epochs=30):
    cfg = VARIANTS[name]
    seed_everything(42)
    run = new_run(f"m3_{name.lower()}", {"variant": name, **{k: cfg[k] for k in cfg}, "quick": quick})
    if cfg["rule"]:
        pack = load_npz("general")
        test = _records(pack, "test_unseen")
        metrics = evaluate_rule(test)
        metrics["variant"] = name
        save_metrics(run, metrics)
        return metrics

    pack = load_npz(cfg["norm"])
    train = _apply_hr(_records(pack, "train"), cfg["hr"])
    val = _apply_hr(_records(pack, "val"), cfg["hr"])
    test = _apply_hr(_records(pack, "test_unseen"), cfg["hr"])
    model = train_model(train, val, epochs=epochs, quick=quick)
    metrics = evaluate_model(model, test)
    metrics["variant"] = name
    # per-session
    per = {}
    for sess in sorted({r["session"] for r in test}):
        sub = [r for r in test if r["session"] == sess]
        per[sess] = evaluate_model(model, sub)
    metrics["per_session"] = per
    # HR ablation note for B4 on w05
    if name == "B4":
        w05 = [r for r in test if r["session"] == "w05"]
        if w05:
            metrics["w05_no_hr"] = evaluate_model(model, w05)
    ckpt = MODELS / f"m3_{name.lower()}.pt"
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"variant": name, "state_dict": model.state_dict(), "in_dim": IN_DIM}, ckpt)
    save_metrics(run, metrics)
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANTS), default="B4")
    p.add_argument("--all", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args(argv)
    if args.rebuild or not (M3_DIR / "sets_general.npz").is_file():
        print(build_all(), flush=True)
    names = list(VARIANTS) if args.all else [args.variant]
    results = [run_variant(n, quick=args.quick, epochs=args.epochs) for n in names]
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
