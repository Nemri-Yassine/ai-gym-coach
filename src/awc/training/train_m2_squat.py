"""Model 2, squat: is this rep correct? Trained on EC3D (lab, 4 people) + Mendeley (gym, 25 people).

Three training conditions, always tested on Mendeley people the model never saw
(5-fold GroupKFold by person, all 3 camera views of a person in the same fold):
  A  EC3D only          (the old Model 2 data; shows the lab→gym gap)
  B  Mendeley only
  C  EC3D + Mendeley
Two models: random forest on hand-made measures + joint-angle summaries, and a binary ST-GCN.
Then the final models (trained on everything of a condition) score our own correct squats,
where the old Model 2 gave mean P(correct) 0.06.

Run: PYTHONPATH=src python -m awc.training.train_m2_squat [--epochs 15] [--no-stgcn]
"""
import argparse
import json
import pickle
import sys
import time
from datetime import date

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn

from awc.config import EXPERIMENTS, MODELS, REPORTS, ROOT
from awc.data.mendeley_form import load_reps
from awc.features.form_dataset import REP_FRAMES, TRAIN_YAWS, load_m2_reps, rep_pose, resample, skeleton_from_rep
from awc.features.form_features import feature_vector, features_2d
from awc.features.skeleton import mirror, normalize
from awc.models.m2_form import STGCNBlock, joint_angles

SEED = 42
N_FOLDS = 5
ANGLE_POINTS = 8  # samples of each angle curve over the rep


# ---------- data ----------

def ec3d_squats():
    """EC3D squat reps at the 7 training yaws, normal and mirrored → (N, 64, 18, 2)."""
    pack = load_m2_reps()
    poses, correct, subject = [], [], []
    for i in np.flatnonzero(pack["exercise"].astype(str) == "squat"):
        p3d = rep_pose(pack, i)
        for yaw in TRAIN_YAWS:
            for mirrored in (False, True):
                poses.append(skeleton_from_rep(p3d, yaw, mirrored))
                correct.append(int(pack["is_correct"][i]))
                subject.append("ec3d_" + str(pack["subject"][i]))
    return np.stack(poses), np.array(correct), np.array(subject)


def with_mirror(poses, y):
    """Train-time augmentation for the 2D gym reps: add the left/right mirror."""
    return np.concatenate([poses, np.stack([mirror(p) for p in poses])]), np.concatenate([y, y])


# ---------- features (random forest) ----------

def rep_features(poses):
    """(N, 64, 18, 2) → (N, 6 + 9 * (4 + ANGLE_POINTS)).

    The six EC3D-style measures (knee width, feet width, depth, lean, knee over toe, sag),
    then for each of the nine joint angles: min, max, mean, range and the curve at 8 points.
    """
    angles = joint_angles(torch.from_numpy(np.asarray(poses, np.float32))).numpy()  # (N, 64, 9)
    idx = np.linspace(0, REP_FRAMES - 1, ANGLE_POINTS).round().astype(int)
    summary = np.concatenate([angles.min(1), angles.max(1), angles.mean(1),
                              angles.max(1) - angles.min(1), angles[:, idx, :].reshape(len(poses), -1)], axis=1)
    measures = np.stack([feature_vector(features_2d(p)) for p in poses])
    return np.nan_to_num(np.concatenate([measures, summary], axis=1))


def fit_rf(x, y):
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                  n_jobs=-1, random_state=SEED).fit(x, y)


# ---------- ST-GCN (binary) ----------

class BinarySTGCN(nn.Module):
    """Same three graph blocks as Model 2's ST-GCN, one logit: P(correct)."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(STGCNBlock(2, 64), STGCNBlock(64, 64), STGCNBlock(64, 128, stride=2))
        self.head = nn.Linear(128, 1)

    def forward(self, x):  # x (B, 64, 18, 2)
        h = self.blocks(x.permute(0, 3, 1, 2))
        return self.head(h.mean(dim=(2, 3))).squeeze(1)


def fit_stgcn(poses, y, epochs, batch=64, lr=1e-3):
    torch.manual_seed(SEED)
    model = BinarySTGCN()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    x = torch.from_numpy(np.asarray(poses, np.float32))
    t = torch.from_numpy(np.asarray(y, np.float32))
    gen = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(x), generator=gen).split(batch):
            xb = x[b] * torch.empty(len(b), 1, 1, 1).uniform_(0.9, 1.1) + 0.01 * torch.randn_like(x[b])
            loss = nn.functional.binary_cross_entropy_with_logits(model(xb), t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def predict_stgcn(model, poses):
    x = torch.from_numpy(np.asarray(poses, np.float32))
    return torch.cat([torch.sigmoid(model(b)) for b in x.split(256)]).numpy()


# ---------- metrics ----------

def scores(y, p, view, video):
    out = {"n": int(len(y)), "balanced_acc": float(balanced_accuracy_score(y, p >= 0.5)),
           "auc": float(roc_auc_score(y, p)),
           "good_called_good": float((p[y == 1] >= 0.5).mean()),
           "bad_called_bad": float((p[y == 0] < 0.5).mean())}
    out["by_view"] = {v: float(balanced_accuracy_score(y[view == v], p[view == v] >= 0.5)) for v in np.unique(view)}
    vids = np.unique(video)
    vy = np.array([y[video == v][0] for v in vids])
    vp = np.array([p[video == v].mean() for v in vids])
    out["video_acc"] = float(((vp >= 0.5) == vy).mean())
    out["n_videos"] = int(len(vids))
    return out


# ---------- own recordings ----------

def own_squat_reps():
    """Our own squat sets (all filmed as correct form) cut into reps the same way as eval_own_recordings."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_own_recordings import activity_span, load_set_pose  # noqa: E402

    from awc.data.own_manifest import load_manifest
    from awc.features.reps import rep_boundaries

    reps, keys = [], []
    for row in load_manifest():
        if row["exercise"] != "squats":
            continue
        try:
            pose, _df = load_set_pose(row)
        except FileNotFoundError:
            continue
        a, b = activity_span(pose)
        seg = pose[a:b]
        finite = seg[np.isfinite(seg).all(axis=(1, 2))]
        if len(finite) < 16:
            continue
        pose_n = normalize(finite)
        for s, e in rep_boundaries(pose_n, exercise="squats"):
            if e - s + 1 >= 8:
                reps.append(resample(pose_n[s:e + 1], REP_FRAMES))
                keys.append(f"{row['person']}/{row['session_folder']}/set{int(row['set']):02d}")
    return np.stack(reps), np.array(keys)


# ---------- main ----------

def run(epochs, use_stgcn):
    t0 = time.perf_counter()
    m = load_reps()
    mx, my, msub, mview, mvideo = m["poses"], m["is_correct"], m["subject"], m["view"], m["video"]
    ex, ey, _esub = ec3d_squats()
    print(f"EC3D squat views {len(ex)} (good {ey.mean():.0%}); Mendeley reps {len(mx)} "
          f"from {len(np.unique(msub))} people", flush=True)

    mf, ef = rep_features(mx), rep_features(ex)
    models = ["rf"] + (["stgcn"] if use_stgcn else [])
    oof = {(c, k): np.full(len(mx), np.nan) for c in "ABC" for k in models}

    # A: EC3D only, one model, tested on every Mendeley rep.
    oof[("A", "rf")][:] = fit_rf(ef, ey).predict_proba(mf)[:, 1]
    if use_stgcn:
        oof[("A", "stgcn")][:] = predict_stgcn(fit_stgcn(ex, ey, epochs), mx)
    print(f"A done {time.perf_counter() - t0:.0f}s", flush=True)

    folds = list(GroupKFold(n_splits=N_FOLDS).split(mx, my, msub))
    for f, (tr, te) in enumerate(folds):
        tr_x, tr_y = with_mirror(mx[tr], my[tr])
        tr_f = rep_features(tr_x)
        oof[("B", "rf")][te] = fit_rf(tr_f, tr_y).predict_proba(mf[te])[:, 1]
        oof[("C", "rf")][te] = fit_rf(np.concatenate([tr_f, ef]), np.concatenate([tr_y, ey])).predict_proba(mf[te])[:, 1]
        if use_stgcn:
            oof[("B", "stgcn")][te] = predict_stgcn(fit_stgcn(tr_x, tr_y, epochs), mx[te])
            oof[("C", "stgcn")][te] = predict_stgcn(
                fit_stgcn(np.concatenate([tr_x, ex]), np.concatenate([tr_y, ey]), epochs), mx[te])
        print(f"fold {f + 1}/{N_FOLDS} done {time.perf_counter() - t0:.0f}s", flush=True)

    results = {}
    for (cond, model), p in oof.items():
        r = scores(my, p, mview, mvideo)
        r["fold_balanced_acc"] = [float(balanced_accuracy_score(my[te], p[te] >= 0.5)) for _tr, te in folds]
        results[f"{cond}_{model}"] = r

    # Shortcut check: can the label be guessed from the first frames alone (standing still,
    # before the squat)? If yes, good and bad videos differ in camera/setup, not only in form.
    static = mx[:, :4].mean(axis=1).reshape(len(mx), -1)
    p_static = np.full(len(mx), np.nan)
    for tr, te in folds:
        p_static[te] = fit_rf(static[tr], my[tr]).predict_proba(static[te])[:, 1]
    results["shortcut_static_first_frames_rf"] = scores(my, p_static, mview, mvideo)

    # Final models on all data → our own correct squats.
    all_x, all_y = with_mirror(mx, my)
    all_f = rep_features(all_x)
    final = {"A_rf": fit_rf(ef, ey), "B_rf": fit_rf(all_f, all_y),
             "C_rf": fit_rf(np.concatenate([all_f, ef]), np.concatenate([all_y, ey]))}
    own_x, own_keys = own_squat_reps()
    own_f = rep_features(own_x)
    own = {}
    for name, rf in final.items():
        p = rf.predict_proba(own_f)[:, 1]
        own[name] = {"n_reps": int(len(p)), "mean_p_correct": float(p.mean()),
                     "pct_called_correct": float((p >= 0.5).mean()),
                     "by_set": {k: round(float(p[own_keys == k].mean()), 3) for k in np.unique(own_keys)}}
    stgcn_final = None
    if use_stgcn:
        stgcn_final = fit_stgcn(np.concatenate([all_x, ex]), np.concatenate([all_y, ey]), epochs)
        p = predict_stgcn(stgcn_final, own_x)
        own["C_stgcn"] = {"n_reps": int(len(p)), "mean_p_correct": float(p.mean()),
                          "pct_called_correct": float((p >= 0.5).mean()),
                          "by_set": {k: round(float(p[own_keys == k].mean()), 3) for k in np.unique(own_keys)}}

    MODELS.mkdir(parents=True, exist_ok=True)
    with open(MODELS / "m2_squat_rf.pickle", "wb") as fh:
        pickle.dump({"forest": final["C_rf"], "features": "rep_features (train_m2_squat.py)",
                     "positive_class": "correct", "trained_on": "EC3D squat + Mendeley squat"}, fh)
    if stgcn_final is not None:
        torch.save(stgcn_final.state_dict(), MODELS / "m2_squat_stgcn.pt")

    out = {"date": str(date.today()), "epochs": epochs, "folds": N_FOLDS, "seconds": round(time.perf_counter() - t0),
           "data": {"mendeley_reps": int(len(mx)), "mendeley_people": int(len(np.unique(msub))),
                    "ec3d_squat_views": int(len(ex))},
           "cv_on_mendeley": results, "own_correct_squats": own}
    run_dir = EXPERIMENTS / f"{date.today():%Y%m%d}_m2_squat_mendeley"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    write_report(out, REPORTS / "m2_squat_mendeley.md")
    print(json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()
                          if kk in ("balanced_acc", "auc", "video_acc", "good_called_good", "bad_called_bad")}
                      for k, v in results.items()}, indent=1))
    print(json.dumps({k: {kk: v[kk] for kk in ("n_reps", "mean_p_correct", "pct_called_correct")}
                      for k, v in own.items()}, indent=1))
    print(f"wrote {run_dir / 'metrics.json'}")
    return out


def write_report(out, path):
    names = {"A": "EC3D only (old data)", "B": "Mendeley only", "C": "EC3D + Mendeley"}
    cv = out["cv_on_mendeley"]
    lines = [
        "# Model 2 · squat good/bad with real gym data (Mendeley)",
        "",
        f"Date {out['date']}. Data: {out['data']['mendeley_reps']} Mendeley squat reps from "
        f"{out['data']['mendeley_people']} people (gym, front/side/diagonal, labels per video) + "
        f"{out['data']['ec3d_squat_views']} EC3D squat rep-views (lab, 4 people, 7 yaws × mirror).",
        f"Test: {out['folds']}-fold cross-validation grouped by person: every number is on Mendeley people "
        "the model never saw. Threshold 0.5. Generated by `python -m awc.training.train_m2_squat`.",
        "",
        "## Unseen gym people (Mendeley)",
        "",
        "| Train data | Model | Balanced acc (rep) | AUC | Good→good | Bad→bad | Video acc | Fold range |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, r in cv.items():
        if key.startswith("shortcut"):
            continue
        cond, model = key.split("_", 1)
        fr = r["fold_balanced_acc"]
        lines.append(f"| {names[cond]} | {model.upper()} | **{r['balanced_acc']:.3f}** | {r['auc']:.3f} | "
                     f"{r['good_called_good']:.2f} | {r['bad_called_bad']:.2f} | {r['video_acc']:.2f} "
                     f"({r['n_videos']}) | {min(fr):.2f}–{max(fr):.2f} |")
    s = cv["shortcut_static_first_frames_rf"]
    lines += [
        "",
        "Balanced acc by camera view (EC3D + Mendeley, RF): "
        + ", ".join(f"{v} {a:.3f}" for v, a in cv["C_rf"]["by_view"].items()),
        "",
        f"**Shortcut check:** a forest that sees only the first 4 frames (standing, before the squat) "
        f"gets balanced acc {s['balanced_acc']:.3f} (AUC {s['auc']:.3f}). Near 0.5 means good and bad videos "
        "are not separable by camera placement or stance alone; well above 0.5 means part of the signal "
        "may be setup, not form.",
        "",
        "## Our own correct squats (old Model 2: mean P(correct) 0.06, 0% called correct)",
        "",
        "| Model | Reps | Mean P(correct) | Called correct |",
        "|---|---:|---:|---:|",
    ]
    for name, r in out["own_correct_squats"].items():
        cond, model = name.split("_", 1)
        lines.append(f"| {names[cond]} · {model.upper()} | {r['n_reps']} | {r['mean_p_correct']:.2f} | "
                     f"{r['pct_called_correct']:.0%} |")
    lines += [
        "",
        "## Limits",
        "",
        "- Mendeley labels are per video: every rep of a bad video counts as bad, so some label noise is expected.",
        "- Binary only (correct / not). The mistake type still comes from EC3D.",
        "- Our own squats are few (all labelled correct), so they test false alarms only, not detection of bad reps.",
        "- Reps are cut by our rep counter; on Mendeley squats the 3 camera angles agree within 2 reps on 70% of sets.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--no-stgcn", action="store_true")
    args = ap.parse_args(argv)
    run(args.epochs, not args.no_stgcn)


if __name__ == "__main__":
    main()
