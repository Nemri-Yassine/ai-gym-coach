"""Model 2 v2: good/bad form on 3-second windows for 7 exercises, trained on real gym people (Mendeley).

Every number is on people the model never saw (5-fold GroupKFold by person; all exercises,
views and both sets of a person stay in the same fold).

Models
  rf, hgb         random forest / gradient boosting on hand-made window features
  rf_cal, hgb_cal same + personal calibration: features minus the mean of the person's
                  K reference windows (start of their GOOD set, same camera view).
                  In the app: the user does a short reference set once, phone fixed.
  tcn             temporal CNN on per-frame joint angles + coordinates (one net, all exercises)
Controls (shortcut checks)
  prop, prop_cal  forest on bone-length ratios only (camera placement / body shape, no movement)

Evaluation set is identical for all models: the reference windows (and the windows overlapping
them) are removed from the test for everyone.

Run: PYTHONPATH=src python -u -m awc.training.train_m2_gym [--epochs 15] [--no-tcn]
"""
import argparse
import json
import pickle
import sys
import time
from datetime import date

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn

from awc.config import EXPERIMENTS, MODELS, REPORTS, ROOT
from awc.data.mendeley_form import load_windows
from awc.features.form_windows import FORM_EXERCISES, angle_sequence, set_windows, window_features
from awc.features.skeleton import normalize

SEED = 42
N_FOLDS = 5
K_REF = 3                      # reference windows (= first ~6 s of the good set)
OWN_TO_FORM = {"squats", "pushups", "bicep_curls", "dumbbell_rows", "dumbbell_shoulder_press", "tricep_extensions"}

T0 = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - T0:6.0f}s] {msg}", flush=True)


# ---------- calibration ----------

def reference_mask(widx, is_correct, k=K_REF):
    """Reference windows: first k windows of a good set. Windows overlapping them (index k) are
    also excluded from evaluation, since hop = half a window."""
    ref = (is_correct == 1) & (widx < k)
    excluded = (is_correct == 1) & (widx <= k)
    return ref, excluded


def calibrate(features, groups, ref):
    """features minus the mean reference features of the same group (person + view).
    Groups without reference windows get zeros for the delta part."""
    delta = np.zeros_like(features)
    for g in np.unique(groups):
        m = groups == g
        r = m & ref
        if r.any():
            delta[m] = features[m] - features[r].mean(axis=0)
    return np.concatenate([features, delta], axis=1)


# ---------- models ----------

THREADS_PER_JOB = 3            # 7 exercises in parallel x 3 threads on a 24-thread CPU


def fit_rf(x, y):
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                                  class_weight="balanced", n_jobs=THREADS_PER_JOB, random_state=SEED).fit(x, y)


def fit_hgb(x, y):
    # 150 rounds, 64 bins: ~4x faster than 300/255 on 2.5k windows, same family of model.
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_bins=64, max_leaf_nodes=31,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=SEED).fit(x, y)


FITTERS = {"rf": fit_rf, "hgb": fit_hgb}


def fold_exercise(a, b, c, feats, feats_cal, props, props_cal, y):
    """All forest/boosting models of one exercise in one fold. Runs in its own worker process.
    a: plain train rows, c: calibrated train rows, b: test rows. Returns {model: P(correct) on b}."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(THREADS_PER_JOB):
        out = {}
        for kind in ("rf", "hgb"):
            out[kind] = FITTERS[kind](feats[a], y[a]).predict_proba(feats[b])[:, 1]
            out[f"{kind}_cal"] = FITTERS[kind](feats_cal[c], y[c]).predict_proba(feats_cal[b])[:, 1]
        out["prop"] = fit_rf(props[a], y[a]).predict_proba(props[b])[:, 1]
        out["prop_cal"] = fit_rf(props_cal[c], y[c]).predict_proba(props_cal[b])[:, 1]
    return b, out


class TCN(nn.Module):
    """Per-frame joint angles + coordinates (+ exercise one-hot) → P(correct). Dilated 1D convs."""

    def __init__(self, c_in, n_ex, width=64, dropout=0.2):
        super().__init__()
        layers, c = [], c_in + n_ex
        for d in (1, 2, 4, 8):
            layers += [nn.Conv1d(c, width, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(width), nn.ReLU(),
                       nn.Dropout(dropout)]
            c = width
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(2 * width, 1)

    def forward(self, x, ex_onehot):
        x = torch.cat([x, ex_onehot[:, :, None].expand(-1, -1, x.shape[2])], dim=1)
        h = self.net(x)
        return self.head(torch.cat([h.mean(2), h.amax(2)], dim=1)).squeeze(1)


def fit_tcn(seq, ex, y, epochs, batch=128, lr=2e-3):
    torch.manual_seed(SEED)
    n_ex = len(FORM_EXERCISES)
    model = TCN(seq.shape[1], n_ex)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(seq) + batch - 1) // batch))
    x, t = torch.from_numpy(seq), torch.from_numpy(y.astype(np.float32))
    oh = torch.eye(n_ex)[torch.from_numpy(ex)]
    gen = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(x), generator=gen).split(batch):
            xb = x[b] * torch.empty(len(b), 1, 1).uniform_(0.9, 1.1) + 0.02 * torch.randn_like(x[b])
            loss = nn.functional.binary_cross_entropy_with_logits(model(xb, oh[b]), t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


@torch.no_grad()
def predict_tcn(model, seq, ex):
    oh = torch.eye(len(FORM_EXERCISES))[torch.from_numpy(ex)]
    x = torch.from_numpy(seq)
    return torch.cat([torch.sigmoid(model(x[i:i + 512], oh[i:i + 512])) for i in range(0, len(x), 512)]).numpy()


# ---------- metrics ----------

def scores(y, p, video):
    out = {"n": int(len(y)), "balanced_acc": float(balanced_accuracy_score(y, p >= 0.5)),
           "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
           "good_called_good": float((p[y == 1] >= 0.5).mean()),
           "bad_called_bad": float((p[y == 0] < 0.5).mean())}
    vids = np.unique(video)
    vy = np.array([y[video == v][0] for v in vids])
    vp = np.array([p[video == v].mean() for v in vids])
    out["video_acc"] = float(((vp >= 0.5) == vy).mean())
    out["n_videos"] = int(len(vids))
    return out


# ---------- own recordings ----------

def own_sets():
    """Our own correct sets of the Mendeley exercises → list of (key, person, exercise, windows)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_own_recordings import load_set_pose  # noqa: E402

    from awc.data.own_manifest import load_manifest

    out = []
    for row in load_manifest():
        if row["exercise"] not in OWN_TO_FORM:
            continue
        try:
            pose, _df = load_set_pose(row)
        except FileNotFoundError:
            continue
        finite = pose[np.isfinite(pose).all(axis=(1, 2))]
        if len(finite) < 100:
            continue
        w = set_windows(normalize(finite))
        if len(w):
            out.append((f"{row['person']}/{row['session_folder']}/set{int(row['set']):02d}",
                        row["person"], row["exercise"], w))
    return out


# ---------- main ----------

def run(epochs, use_tcn):
    torch.set_num_threads(12)
    d = load_windows()
    feats, props = d["features"], d["proportions"]
    y, ex_name, subj, view, video, widx = (d["is_correct"], d["exercise"], d["subject"], d["view"],
                                           d["video"], d["widx"])
    ex_id = np.array([FORM_EXERCISES.index(e) for e in ex_name])
    groups = np.char.add(np.char.add(subj, "|"), view)       # calibration group = person + camera view
    ref, excluded = reference_mask(widx, y)
    evaluate = ~excluded
    feats_cal = calibrate(feats, groups, ref)
    props_cal = calibrate(props, groups, ref)
    log(f"windows {len(y)}, people {len(np.unique(subj))}, features {feats.shape[1]}; "
        f"evaluated windows {int(evaluate.sum())}")

    folds = list(GroupKFold(n_splits=N_FOLDS).split(feats, y, subj))
    names = ["rf", "hgb", "rf_cal", "hgb_cal", "prop", "prop_cal"] + (["tcn"] if use_tcn else [])
    oof = {n: np.full(len(y), np.nan) for n in names}

    from joblib import Parallel, delayed

    for f, (tr, te) in enumerate(folds):
        tr_cal = tr[~ref[tr]]                           # calibrated: references are the baseline, not samples
        jobs = [delayed(fold_exercise)(tr[ex_id[tr] == e], te[ex_id[te] == e], tr_cal[ex_id[tr_cal] == e],
                                       feats, feats_cal, props, props_cal, y)
                for e in range(len(FORM_EXERCISES))]
        for b, preds in Parallel(n_jobs=len(FORM_EXERCISES))(jobs):
            for kind, p in preds.items():
                oof[kind][b] = p
        log(f"fold {f + 1}/{N_FOLDS}: forests + boosting done (7 exercises in parallel)")
        if use_tcn:
            seq_tr = angle_sequence(d["windows"][tr].astype(np.float32))
            model = fit_tcn(seq_tr, ex_id[tr], y[tr], epochs)
            del seq_tr
            oof["tcn"][te] = predict_tcn(model, angle_sequence(d["windows"][te].astype(np.float32)), ex_id[te])
            log(f"fold {f + 1}/{N_FOLDS}: temporal CNN done")

    # Out-of-fold predictions: lets ensembles, thresholds and within-person comparisons be
    # tested later without retraining.
    oof_dir = EXPERIMENTS / f"{date.today():%Y%m%d}_m2_gym_windows"
    oof_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(oof_dir / "oof.npz", y=y, exercise=ex_name, subject=subj, view=view, video=video,
                        widx=widx, evaluate=evaluate, fold=np.concatenate(
                            [np.full(len(te), f) for f, (_tr, te) in enumerate(folds)])[np.argsort(
                                np.concatenate([te for _tr, te in folds]))],
                        **{f"p_{k}": v for k, v in oof.items()})

    results = {}
    for n, p in oof.items():
        per_ex = {}
        for e, exn in enumerate(FORM_EXERCISES):
            m = evaluate & (ex_id == e)
            per_ex[exn] = scores(y[m], p[m], video[m])
            per_ex[exn]["fold_balanced_acc"] = [
                float(balanced_accuracy_score(y[np.intersect1d(te, np.flatnonzero(m))],
                                              p[np.intersect1d(te, np.flatnonzero(m))] >= 0.5)) for _tr, te in folds]
        results[n] = {"mean_balanced_acc": float(np.mean([v["balanced_acc"] for v in per_ex.values()])),
                      "mean_video_acc": float(np.mean([v["video_acc"] for v in per_ex.values()])),
                      "per_exercise": per_ex}
        log(f"{n:9s} mean balanced acc {results[n]['mean_balanced_acc']:.3f} | "
            + " ".join(f"{k[:8]} {v['balanced_acc']:.2f}" for k, v in per_ex.items()))

    # ---- final models (all people) + own recordings ----
    best_plain = max(("rf", "hgb"), key=lambda k: results[k]["mean_balanced_acc"])
    best_cal = max(("rf_cal", "hgb_cal"), key=lambda k: results[k]["mean_balanced_acc"])
    final = {"plain": {}, "cal": {}}
    for e, exn in enumerate(FORM_EXERCISES):
        m = ex_id == e
        final["plain"][exn] = FITTERS[best_plain](feats[m], y[m])
        mc = m & ~ref
        final["cal"][exn] = FITTERS[best_cal.split("_")[0]](feats_cal[mc], y[mc])
    log(f"final models trained ({best_plain}, {best_cal})")

    own = []
    for key, person, exn, w in own_sets():
        f_own = window_features(w)
        p_plain = final["plain"][exn].predict_proba(f_own)[:, 1]
        own.append({"key": key, "person": person, "exercise": exn, "windows": int(len(w)),
                    "p_plain": p_plain.tolist(), "_w": w})
    # calibration on our data: reference = first K windows of the person's first set of that exercise
    for (person, exn) in sorted({(o["person"], o["exercise"]) for o in own}):
        sets = [o for o in own if o["person"] == person and o["exercise"] == exn]
        ref_feats = window_features(sets[0]["_w"][:K_REF])
        for o in sets:
            w = o["_w"]
            f_own = window_features(w)
            delta = f_own - ref_feats.mean(axis=0)
            p_cal = final["cal"][exn].predict_proba(np.concatenate([f_own, delta], axis=1))[:, 1]
            keep = np.ones(len(w), bool)
            if o is sets[0]:
                keep[:K_REF + 1] = False          # reference windows are not scored
            o["p_cal"] = p_cal[keep].tolist()
    own_summary = {}
    for kind in ("p_plain", "p_cal"):
        allp = np.concatenate([np.array(o[kind]) for o in own if len(o.get(kind, []))])
        own_summary[kind] = {
            "windows": int(len(allp)), "mean_p_correct": float(allp.mean()),
            "called_correct": float((allp >= 0.5).mean()),
            "sets_called_correct": float(np.mean([np.mean(o[kind]) >= 0.5 for o in own if len(o.get(kind, []))])),
            "by_exercise": {exn: float(np.mean(np.concatenate([np.array(o[kind]) >= 0.5 for o in own
                                                                if o["exercise"] == exn and len(o.get(kind, []))])))
                            for exn in sorted({o["exercise"] for o in own})},
        }
    log("own recordings: " + json.dumps({k: {kk: round(vv, 3) if isinstance(vv, float) else vv
                                              for kk, vv in v.items() if kk != "by_exercise"}
                                          for k, v in own_summary.items()}))

    MODELS.mkdir(parents=True, exist_ok=True)
    with open(MODELS / "m2_gym.pickle", "wb") as fh:
        pickle.dump({"version": 2, "window_s": 3.0, "k_ref": K_REF, "exercises": FORM_EXERCISES,
                     "plain": final["plain"], "cal": final["cal"], "plain_kind": best_plain,
                     "cal_kind": best_cal, "positive_class": "correct",
                     "features": "awc.features.form_windows.window_features"}, fh)

    out = {"date": str(date.today()), "folds": N_FOLDS, "k_ref": K_REF, "epochs": epochs,
           "seconds": round(time.perf_counter() - T0),
           "data": {"windows": int(len(y)), "evaluated": int(evaluate.sum()), "people": int(len(np.unique(subj)))},
           "cv": results, "own": own_summary, "own_sets": [{k: v for k, v in o.items() if not k.startswith(("p_", "_"))}
                                                           | {"mean_p_plain": float(np.mean(o["p_plain"])),
                                                              "mean_p_cal": float(np.mean(o["p_cal"])) if o.get("p_cal") else None}
                                                           for o in own],
           "best_plain": best_plain, "best_cal": best_cal}
    run_dir = EXPERIMENTS / f"{date.today():%Y%m%d}_m2_gym_windows"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    log(f"wrote {run_dir / 'metrics.json'}")
    return out


LABELS = {"rf": "Random forest", "hgb": "Gradient boosting", "rf_cal": "Random forest + personal calibration",
          "hgb_cal": "Gradient boosting + personal calibration", "tcn": "Temporal CNN (deep learning)",
          "prop": "Control: body proportions only", "prop_cal": "Control: body proportions + calibration"}
SHORT = {"squats": "Squat", "pushups": "Push-up", "bicep_curls": "Curl", "dumbbell_rows": "Row",
         "dumbbell_shoulder_press": "Sh. press", "tricep_extensions": "Triceps", "side_bends": "Side bend"}


def write_report(out, path=REPORTS / "m2_gym_windows.md"):
    cv = out["cv"]
    exs = list(next(iter(cv.values()))["per_exercise"])
    lines = [
        "# Model 2 v2 · good/bad form on real gym people, 7 exercises",
        "",
        f"Date {out['date']}. Data: Mendeley multi-view dataset, {out['data']['people']} people, "
        f"{out['data']['windows']} windows of 3 s (hop 1.5 s), one good and one bad set per person and exercise, "
        "front/side/diagonal. Labels per set.",
        f"Test: {out['folds']}-fold cross-validation grouped by person (unseen people). The same "
        f"{out['data']['evaluated']} windows are scored for every model: the {out['k_ref']} reference windows "
        "at the start of each good set (and the one overlapping them) are left out for all.",
        "Generated by `python -m awc.training.train_m2_gym`; numbers in "
        f"`experiments/{out['date'].replace('-', '')}_m2_gym_windows/metrics.json`.",
        "",
        "## Balanced accuracy per window, unseen people",
        "",
        "| Model | Mean | " + " | ".join(SHORT.get(e, e) for e in exs) + " | Set acc (mean) |",
        "|---|---:|" + "---:|" * len(exs) + "---:|",
    ]
    order = [k for k in ("rf", "hgb", "tcn", "rf_cal", "hgb_cal", "prop", "prop_cal") if k in cv]
    for k in order:
        r = cv[k]
        lines.append(f"| {LABELS[k]} | **{r['mean_balanced_acc']:.3f}** | "
                     + " | ".join(f"{r['per_exercise'][e]['balanced_acc']:.2f}" for e in exs)
                     + f" | {r['mean_video_acc']:.2f} |")
    lines += ["", "AUC per exercise (best plain model "
              f"`{out['best_plain']}`): " + ", ".join(
                  f"{SHORT.get(e, e)} {cv[out['best_plain']]['per_exercise'][e]['auc']:.2f}" for e in exs), ""]
    own = out["own"]
    lines += [
        "## Our own correct sets (false alarms)",
        "",
        "All our recorded sets are correct form, so this measures how often Model 2 wrongly says 'bad'. "
        "Calibration uses the first 3 windows of the person's first set of that exercise as reference "
        "(not scored).",
        "",
        "| Model | Windows | Windows called correct | Sets called correct | " +
        " | ".join(SHORT.get(e, e) for e in own["p_plain"]["by_exercise"]) + " |",
        "|---|---:|---:|---:|" + "---:|" * len(own["p_plain"]["by_exercise"]),
    ]
    for kind, label in (("p_plain", f"Plain ({out['best_plain']})"), ("p_cal", f"Calibrated ({out['best_cal']})")):
        o = own[kind]
        lines.append(f"| {label} | {o['windows']} | **{o['called_correct']:.0%}** | {o['sets_called_correct']:.0%} | "
                     + " | ".join(f"{v:.0%}" for v in o["by_exercise"].values()) + " |")
    lines += ["", "Per set: " + "; ".join(
        f"{s['key'].split('/')[0]} {SHORT.get(s['exercise'], s['exercise'])} {s['mean_p_plain']:.2f}"
        + (f"/{s['mean_p_cal']:.2f}" if s.get("mean_p_cal") is not None else "") for s in out["own_sets"]),
        "(mean P(correct), plain/calibrated)", ""]
    rel_path = EXPERIMENTS / f"{out['date'].replace('-', '')}_m2_gym_windows" / "relative.json"
    if rel_path.is_file():
        rel = json.loads(rel_path.read_text())
        lines += [
            "## Absolute vs relative judgement (within the same person)",
            "",
            "Absolute: is this set bad? Relative: of one person's two sets (same exercise, same camera), "
            "which is worse? Both sets are separate recordings, so nothing leaks from a shared recording. "
            "From `scripts/analysis/m2_relative.py` on the same out-of-fold predictions.",
            "",
            "| Model | Absolute set acc | Relative pair acc | Pairs |",
            "|---|---:|---:|---:|",
        ]
        for r in rel["summary"]:
            label = LABELS.get(r["model"], "Ensemble (mean of plain models)" if r["model"] == "ensemble" else r["model"])
            lines.append(f"| {label} | {r['absolute_set_acc']:.3f} | **{r['relative_pair_acc']:.3f}** | {r['pairs']} |")
        ens = rel["per_exercise"].get("ensemble", {})
        lines += ["", "Relative accuracy per exercise (ensemble): "
                  + ", ".join(f"{SHORT.get(e, e)} {v:.2f}" for e, v in ens.items()), ""]
    ens_path = rel_path.parent / "ensemble.json"
    if ens_path.is_file():
        e = json.loads(ens_path.read_text())
        lines += [
            "## Live verdict: ensemble, 'worse than your usual'",
            "",
            "The app scores each set with the mean of RF + boosting + temporal CNN (all trained on every "
            "Mendeley person) and flags it when it is at least the drop below the person's usual (median of "
            "earlier sets of the same exercise).",
            "",
            "| Drop | Bad sets caught (Mendeley pairs) | Our correct sets flagged (own pairs) |",
            "|---:|---:|---:|",
        ]
        for t in e["tradeoff"]:
            lines.append(f"| {t['drop']:.2f} | {t['mendeley_bad_caught']:.0%} | "
                         f"{t['own_correct_flagged']:.0%} ({t['own_pairs']}) |")
        lines += ["", f"Chosen drop: 0.10. Absolute verdict with the ensemble: Mendeley set acc "
                  f"{e['absolute']['mendeley_set_acc']:.3f}, our correct sets called correct "
                  f"{e['absolute']['own_sets_called_correct']:.0%}. Own pairs are few (18): one pair = 5.6 points.",
                  ""]
    lines += [
        "## Reading (hand-written summary, 2026-11-21)",
        "",
        "- **Coverage:** Model 2 now judges 7 exercises on real gym people (before: squat only). Mean "
        "balanced accuracy is about 0.68 per window on unseen people. Triceps is the best (0.75), rows the weakest (0.60).",
        "- **Set accuracy is about the same as window accuracy (0.69).** Errors are whole people, not "
        "random moments: one absolute line does not fit 26 different bodies and setups.",
        "- **Relative judgement is the fix:** ranking a person's two sets is right 81% of the time with the "
        "best single model and **85% with the ensemble** (RF + boosting + temporal CNN; control on body "
        "proportions only: 0.58). The live demo compares each set with the person's usual and flags a "
        "drop of 0.10 or more: bad sets caught 70%, our own correct sets flagged 28% (18 pairs).",
        "- **Temporal CNN (deep learning):** alone it ties the forests (0.672 per window, best set accuracy "
        "0.695, best on rows), but its mistakes differ, so it lifts the ensemble from 0.81 to 0.85 "
        "on the relative question.",
        "- **Personal calibration failed (leak):** features minus a reference from the start of the good "
        "set looked +1.5 points better, but reference and good test windows share one recording. On our "
        "own new sets it called only 23% correct. Switched off (`USE_CALIBRATION = False`).",
        "- **Shortcut control:** body proportions alone give 0.54 (absolute) and 0.58 (relative), so the "
        "models mostly learn movement, not camera placement.",
        "- **Open problems:** (1) a fair calibration needs two good sets of the same person, which Mendeley "
        "does not have; (2) no bad-form recording of our own yet, so detection on our camera is untested; "
        "(3) mistake type (knees in, not low…) still only from EC3D.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--no-tcn", action="store_true")
    ap.add_argument("--only-tcn", action="store_true", help="train/evaluate only the temporal CNN, merge into metrics")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--final-ensemble", action="store_true",
                    help="train RF + boosting + temporal CNN on everyone, add to models/m2_gym.pickle")
    ap.add_argument("--eval-ensemble", action="store_true", help="only re-measure the saved ensemble")
    args = ap.parse_args(argv)
    # Latest run with out-of-fold predictions (a run can cross midnight).
    done = sorted(p for p in EXPERIMENTS.glob("*_m2_gym_windows") if (p / "oof.npz").exists())
    run_dir = done[-1] if done else EXPERIMENTS / f"{date.today():%Y%m%d}_m2_gym_windows"
    if args.final_ensemble:
        final_ensemble(args.epochs, run_dir)
        return
    if args.eval_ensemble:
        evaluate_ensemble(run_dir)
        return
    if args.report_only:
        print(write_report(json.loads((run_dir / "metrics.json").read_text())))
        return
    if args.only_tcn:
        run_tcn_only(args.epochs, run_dir)
        return
    out = run(args.epochs, not args.no_tcn)
    print(write_report(out))


def final_ensemble(epochs, run_dir):
    """Final ensemble for the app: per-exercise RF + boosting, one temporal CNN, all trained on every
    Mendeley person. Mean of the three = P(correct). Then false alarms on our own correct sets and
    the relative-drop trade-off are recomputed with the ensemble."""
    import itertools

    from joblib import Parallel, delayed

    torch.set_num_threads(16)
    d = load_windows()
    feats, y, ex_name = d["features"], d["is_correct"], d["exercise"]
    ex_id = np.array([FORM_EXERCISES.index(e) for e in ex_name])

    def fit_pair(e):
        from threadpoolctl import threadpool_limits

        m = ex_id == e
        with threadpool_limits(THREADS_PER_JOB):
            return FORM_EXERCISES[e], fit_rf(feats[m], y[m]), fit_hgb(feats[m], y[m])

    forests = {name: (rf, hgb) for name, rf, hgb in
               Parallel(n_jobs=len(FORM_EXERCISES))(delayed(fit_pair)(e) for e in range(len(FORM_EXERCISES)))}
    log("final forests + boosting trained")
    seq = angle_sequence(d["windows"].astype(np.float32))
    tcn = fit_tcn(seq, ex_id, y, epochs)
    del seq
    log("final temporal CNN trained")

    path = MODELS / "m2_gym.pickle"
    with open(path, "rb") as fh:
        pack = pickle.load(fh)
    pack["ensemble"] = {"forests": forests, "tcn_state": tcn.state_dict(), "tcn_channels": 45,
                        "members": ["rf", "hgb", "tcn"], "epochs": epochs}
    with open(path, "wb") as fh:
        pickle.dump(pack, fh)
    log(f"saved ensemble into {path}")
    evaluate_ensemble(run_dir)


def evaluate_ensemble(run_dir):
    """False alarms on our own correct sets and the relative-drop trade-off, with the saved ensemble."""
    import itertools

    import pandas as pd

    from awc.models.m2_gym import ensemble_proba, load_model

    load_model.cache_clear()
    pack = load_model()

    own = []
    for key, person, exn, w in own_sets():
        own.append({"key": key, "person": person, "exercise": exn,
                    "p": float(ensemble_proba(pack, exn, w).mean())})
    z = np.load(run_dir / "oof.npz", allow_pickle=False)
    p_ens = np.mean([z["p_rf"], z["p_hgb"], np.load(run_dir / "oof_tcn.npy")], axis=0)
    ev = z["evaluate"]
    sets = (pd.DataFrame({"subject": z["subject"][ev], "exercise": z["exercise"][ev], "view": z["view"][ev],
                          "video": z["video"][ev], "y": z["y"][ev], "p": p_ens[ev]})
            .groupby(["subject", "exercise", "view", "video"], as_index=False).mean(numeric_only=True))
    pairs = np.array([(g[g.y == 1].p.iloc[0], g[g.y == 0].p.iloc[0])
                      for _k, g in sets.groupby(["subject", "exercise", "view"]) if set(g.y) == {0.0, 1.0}])
    own_df = pd.DataFrame(own)
    own_pairs = np.array([(a, b) for (_pp, _ee), g in own_df.groupby(["person", "exercise"])
                          for a, b in itertools.permutations(g.p.values, 2)])
    tradeoff = []
    for drop in (0.0, 0.05, 0.10, 0.15, 0.20):
        tradeoff.append({"drop": drop, "mendeley_bad_caught": float((pairs[:, 1] < pairs[:, 0] - drop).mean()),
                         "own_correct_flagged": float((own_pairs[:, 1] < own_pairs[:, 0] - drop).mean()),
                         "own_pairs": int(len(own_pairs))})
        log(f"drop {drop:.2f}: bad caught {tradeoff[-1]['mendeley_bad_caught']:.0%}, "
            f"own correct flagged {tradeoff[-1]['own_correct_flagged']:.0%} ({len(own_pairs)} pairs)")
    absolute = {"own_sets_called_correct": float((own_df.p >= 0.5).mean()),
                "mendeley_set_acc": float(((sets.p >= 0.5).astype(float) == sets.y).mean())}
    log(f"absolute with ensemble: own correct sets called correct {absolute['own_sets_called_correct']:.0%}, "
        f"Mendeley set acc {absolute['mendeley_set_acc']:.3f}")
    (run_dir / "ensemble.json").write_text(json.dumps(
        {"tradeoff": tradeoff, "absolute": absolute, "own_sets": own}, indent=2))


def run_tcn_only(epochs, run_dir):
    """Temporal CNN alone (same folds and evaluated windows), merged into an existing metrics.json."""
    torch.set_num_threads(16)
    d = load_windows()
    y, ex_name, subj, video, widx = d["is_correct"], d["exercise"], d["subject"], d["video"], d["widx"]
    ex_id = np.array([FORM_EXERCISES.index(e) for e in ex_name])
    _ref, excluded = reference_mask(widx, y)
    evaluate = ~excluded
    folds = list(GroupKFold(n_splits=N_FOLDS).split(d["features"], y, subj))
    p = np.full(len(y), np.nan)
    for f, (tr, te) in enumerate(folds):
        model = fit_tcn(angle_sequence(d["windows"][tr].astype(np.float32)), ex_id[tr], y[tr], epochs)
        p[te] = predict_tcn(model, angle_sequence(d["windows"][te].astype(np.float32)), ex_id[te])
        log(f"tcn fold {f + 1}/{N_FOLDS} done")
    per_ex = {}
    for e, exn in enumerate(FORM_EXERCISES):
        m = evaluate & (ex_id == e)
        per_ex[exn] = scores(y[m], p[m], video[m])
    res = {"mean_balanced_acc": float(np.mean([v["balanced_acc"] for v in per_ex.values()])),
           "mean_video_acc": float(np.mean([v["video_acc"] for v in per_ex.values()])), "per_exercise": per_ex}
    log(f"tcn mean balanced acc {res['mean_balanced_acc']:.3f} | "
        + " ".join(f"{k[:8]} {v['balanced_acc']:.2f}" for k, v in per_ex.items()))
    np.save(run_dir / "oof_tcn.npy", p)
    path = run_dir / "metrics.json"
    out = json.loads(path.read_text())
    out["cv"]["tcn"] = res
    out["epochs"] = epochs
    path.write_text(json.dumps(out, indent=2))
    print(write_report(out))


if __name__ == "__main__":
    main()
