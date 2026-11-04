"""Train Model 2 and compare ST-GCN, the angle LSTM and a random forest.

    python -m awc.training.train_m2 --model stgcn
    python -m awc.training.train_m2 --model lstm
    python -m awc.training.train_m2 --model rf
    python -m awc.training.train_m2 --model stgcn --quick

Four leave-one-person-out folds, fixed epochs, no early stopping.
The scored test is the 45° view. The same reps are scored again at every
yaw from 0° to 90°. A final checkpoint is trained on all four people for
live use and is not scored.
"""
import argparse
import pickle
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from awc.config import FIGURES, M2_REPS, MODELS, REPORTS
from awc.evaluation.metrics import accuracy
from awc.features.form_dataset import (
    EXERCISES,
    N_MISTAKES,
    REP_FRAMES,
    TEST_YAW,
    TRAIN_YAWS,
    augment_skeleton,
    fold_indices,
    load_m2_reps,
    loso_folds,
    rep_pose,
    skeleton_from_rep,
)
from awc.features.form_features import feature_vector, features_2d, features_3d
from awc.models.m2_form import build_model
from awc.training.tracking import new_run, save_metrics, seed_everything

BATCH = 32
RF_TREES = 300
# Cited figures. They are not measurements from this benchmark.
CONTEXT_AUTHORS = 0.909
CONTEXT_EARLIER_SQUAT_RF = 0.53


def _log(run_dir, message):
    print(message, flush=True)
    with (run_dir / "log.txt").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _indices_for_quick(pack):
    exercise = np.asarray(pack["exercise"]).astype(str)
    subject = np.asarray(pack["subject"]).astype(str)
    chosen = []
    for person in sorted(set(subject.tolist())):
        for name in EXERCISES:
            hit = np.flatnonzero((subject == person) & (exercise == name))
            if len(hit):
                chosen.append(int(hit[0]))
    return np.asarray(chosen, dtype=np.int64)


def _slice_pack(pack, indices):
    out = {}
    for key, value in pack.items():
        out[key] = value[indices]
    return out


def load_pack(quick):
    if not M2_REPS.exists():
        raise FileNotFoundError(f"{M2_REPS} is missing. Build it with python -m awc.features.form_dataset")
    pack = load_m2_reps()
    if quick:
        pack = _slice_pack(pack, _indices_for_quick(pack))
    return pack


def project_cache(pack):
    """Every rep at every train yaw, normal and mirrored. (N, yaw, mirror, 64, 18, 2)."""
    n = len(pack["exercise"])
    cache = np.empty((n, len(TRAIN_YAWS), 2, REP_FRAMES, 18, 2), np.float32)
    started = time.perf_counter()
    for i in range(n):
        pose3d = rep_pose(pack, i)
        for yaw_i, yaw in enumerate(TRAIN_YAWS):
            for mirror_i, mirrored in enumerate((False, True)):
                cache[i, yaw_i, mirror_i] = skeleton_from_rep(pose3d, yaw, mirrored)
    return cache, time.perf_counter() - started


def _labels(pack):
    names = {name: i for i, name in enumerate(EXERCISES)}
    exercise = np.array([names[str(v)] for v in pack["exercise"]], dtype=np.int64)
    return exercise, pack["mistake"].astype(np.int64), pack["is_correct"].astype(np.int64)


def _take(cache, rep_index, yaw_index, mirror):
    return cache[rep_index, yaw_index, mirror]


class _RepDataset(Dataset):
    """Training views. Augmentation uses a per-sample generator. num_workers stays 0."""

    def __init__(self, poses, exercise, mistake, augment, seed):
        self.poses = poses
        self.exercise = np.asarray(exercise, dtype=np.int64)
        self.mistake = np.asarray(mistake, dtype=np.int64)
        self.augment = augment
        self.seed = int(seed)
        self._calls = 0

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        pose = self.poses[index]
        if self.augment:
            self._calls += 1
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, 0, int(index), self._calls])
            )
            pose = augment_skeleton(pose, rng)
        x = torch.from_numpy(np.ascontiguousarray(pose, dtype=np.float32))
        return x.permute(2, 0, 1), int(self.exercise[index]), int(self.mistake[index])


def _stack_views(cache, rep_ids, exercise, mistake, yaw_indexes, mirrors):
    poses, ex, y = [], [], []
    for rep in rep_ids:
        for yaw_i in yaw_indexes:
            for mirror in mirrors:
                poses.append(_take(cache, int(rep), int(yaw_i), int(mirror)))
                ex.append(int(exercise[rep]))
                y.append(int(mistake[rep]))
    if not poses:
        empty = np.zeros((0, REP_FRAMES, 18, 2), np.float32)
        return empty, np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.stack(poses), np.asarray(ex, np.int64), np.asarray(y, np.int64)


def _head_loss(logits, exercise, target):
    total = None
    seen = 0
    for index, name in enumerate(EXERCISES):
        mask = exercise == index
        if not mask.any():
            continue
        part = nn.functional.cross_entropy(logits[name][mask], target[mask], reduction="sum")
        total = part if total is None else total + part
        seen += int(mask.sum())
    if total is None:
        raise RuntimeError("empty batch")
    return total / seen


def _predict_net(model, poses, exercise, batch):
    model.eval()
    pred = np.empty(len(poses), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(poses), batch):
            stop = start + batch
            x = torch.from_numpy(poses[start:stop]).permute(0, 3, 1, 2)
            ex = torch.from_numpy(exercise[start:stop])
            logits = model(x, ex)
            chosen = torch.empty(ex.shape[0], dtype=torch.long)
            for index, name in enumerate(EXERCISES):
                mask = ex == index
                if mask.any():
                    chosen[mask] = logits[name][mask].argmax(dim=1)
            pred[start:stop] = chosen.numpy()
    return pred


def _fit_epoch(model, loader, opt):
    model.train()
    total, seen = 0.0, 0
    for pose, exercise, mistake in loader:
        opt.zero_grad()
        loss = _head_loss(model(pose, exercise), exercise, mistake)
        loss.backward()
        opt.step()
        total += float(loss.item()) * len(mistake)
        seen += len(mistake)
    return total / max(seen, 1)


def _majority(labels):
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) == 0:
        return 0
    return int(np.bincount(labels).argmax())


def _binary_mistake(is_correct, pred):
    """Positive class = not correct. Specificity is correct reps kept as correct."""
    is_correct = np.asarray(is_correct, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    true_mistake = is_correct == 0
    pred_mistake = pred != 0
    tp = int((true_mistake & pred_mistake).sum())
    fp = int((~true_mistake & pred_mistake).sum())
    tn = int((~true_mistake & ~pred_mistake).sum())
    fn = int((true_mistake & ~pred_mistake).sum())

    def _div(num, den):
        return None if den == 0 else num / den

    return {
        "mistake_recall": _div(tp, tp + fn),
        "mistake_precision": _div(tp, tp + fp),
        "mistake_specificity": _div(tn, tn + fp),
    }


def _score_exercise(y_true, y_pred, is_correct, n_classes, majority_class):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    is_correct = np.asarray(is_correct, dtype=np.int64)
    n = int(len(y_true))
    if n == 0:
        return {
            "n": 0, "n_correct": 0, "accuracy": None, "macro_f1": None,
            "majority_class": int(majority_class), "majority_accuracy": None,
            "mistake_recall": None, "mistake_precision": None, "mistake_specificity": None,
            "always_mistake_recall": None, "always_mistake_precision": None,
            "always_mistake_specificity": None,
        }
    binary = _binary_mistake(is_correct, y_pred)
    always = _binary_mistake(is_correct, np.ones(n, dtype=np.int64))
    majority_pred = np.full(n, int(majority_class), dtype=np.int64)
    return {
        "n": n,
        "n_correct": int((is_correct == 1).sum()),
        "accuracy": float(accuracy(y_true, y_pred)),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0,
        )),
        "majority_class": int(majority_class),
        "majority_accuracy": float(accuracy(y_true, majority_pred)),
        **binary,
        "always_mistake_recall": always["mistake_recall"],
        "always_mistake_precision": always["mistake_precision"],
        "always_mistake_specificity": always["mistake_specificity"],
    }


def _score_predictions(y_true, y_pred, exercise, is_correct, majority):
    per = {}
    for index, name in enumerate(EXERCISES):
        mask = exercise == index
        per[name] = _score_exercise(
            y_true[mask], y_pred[mask], is_correct[mask], N_MISTAKES[name], majority[name],
        )
    overall = None if len(y_true) == 0 else float(accuracy(y_true, y_pred))
    return {"per_exercise": per, "overall_accuracy": overall}


def _mean_std(values):
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None
    arr = np.asarray(clean, dtype=np.float64)
    std = 0.0 if len(arr) == 1 else float(arr.std(ddof=1))
    return float(arr.mean()), std


def summarize_folds(folds):
    keys = (
        "accuracy", "macro_f1", "mistake_recall", "mistake_precision", "mistake_specificity",
        "majority_accuracy", "always_mistake_recall", "always_mistake_precision",
        "always_mistake_specificity",
    )
    per = {}
    for name in EXERCISES:
        cell = {}
        for key in keys:
            mean, std = _mean_std([fold["yaw_45"]["per_exercise"][name][key] for fold in folds])
            cell[f"{key}_mean"] = mean
            cell[f"{key}_std"] = std
        cell["n_mean"] = _mean_std([fold["yaw_45"]["per_exercise"][name]["n"] for fold in folds])[0]
        per[name] = cell
    angles = {}
    for yaw in (str(v) for v in TRAIN_YAWS):
        angles[yaw] = {}
        for name in EXERCISES:
            mean, std = _mean_std([
                fold["angles"][yaw]["per_exercise"][name]["accuracy"] for fold in folds
            ])
            recall, recall_std = _mean_std([
                fold["angles"][yaw]["per_exercise"][name]["mistake_recall"] for fold in folds
            ])
            angles[yaw][name] = {
                "accuracy_mean": mean, "accuracy_std": std,
                "mistake_recall_mean": recall, "mistake_recall_std": recall_std,
            }
        mean, std = _mean_std([fold["angles"][yaw]["overall_accuracy"] for fold in folds])
        angles[yaw]["overall"] = {"accuracy_mean": mean, "accuracy_std": std}
    return {"per_exercise": per, "angles": angles}


def _correct_share(exercise, is_correct):
    out = {}
    for index, name in enumerate(EXERCISES):
        mask = exercise == index
        out[name] = None if not mask.any() else float(is_correct[mask].mean())
    return out


def _feature_matrix(poses):
    rows = [feature_vector(features_2d(pose)) for pose in poses]
    if not rows:
        return np.zeros((0, 6), np.float64)
    return np.vstack(rows)


def _fit_forests(features, mistake, exercise, seed):
    forests = {}
    for index, name in enumerate(EXERCISES):
        mask = exercise == index
        if not mask.any():
            continue
        forests[name] = RandomForestClassifier(
            n_estimators=RF_TREES, random_state=seed, n_jobs=1,
        ).fit(features[mask], mistake[mask])
    return forests


def _predict_forests(forests, features, exercise):
    pred = np.zeros(len(features), dtype=np.int64)
    for index, name in enumerate(EXERCISES):
        mask = exercise == index
        if not mask.any():
            continue
        if name not in forests:
            pred[mask] = 0
            continue
        pred[mask] = forests[name].predict(features[mask])
    return pred


def _train_folds_net(model_name, pack, cache, exercise, mistake, is_correct, epochs, batch, lr, seed, run_dir):
    folds_out = []
    train_yaws = list(range(len(TRAIN_YAWS)))
    started = time.perf_counter()
    for fold in loso_folds(pack["subject"]):
        seed_everything(seed)
        train_ids, test_ids = fold_indices(pack["subject"], fold["test"])
        majority = {
            name: _majority(mistake[train_ids][exercise[train_ids] == index])
            for index, name in enumerate(EXERCISES)
        }
        poses, ex, y = _stack_views(cache, train_ids, exercise, mistake, train_yaws, (0, 1))
        loader = DataLoader(
            _RepDataset(poses, ex, y, augment=True, seed=seed),
            batch_size=batch, shuffle=True, num_workers=0,
        )
        model = build_model(model_name)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            loss = _fit_epoch(model, loader, opt)
            _log(run_dir, f"fold {fold['fold'] + 1} {fold['test']} epoch {epoch + 1}/{epochs} loss {loss:.4f}")
        angles = {}
        is_corr = is_correct
        for yaw_i, yaw in enumerate(TRAIN_YAWS):
            view, view_ex, _ = _stack_views(cache, test_ids, exercise, mistake, [yaw_i], (0,))
            pred = _predict_net(model, view, view_ex, batch)
            angles[str(yaw)] = _score_predictions(
                mistake[test_ids], pred, exercise[test_ids], is_corr[test_ids], majority,
            )
        folds_out.append({
            "fold": int(fold["fold"]),
            "test_subject": fold["test"],
            "n_train_reps": int(len(train_ids)),
            "n_test_reps": int(len(test_ids)),
            "yaw_45": angles[str(TEST_YAW)],
            "angles": angles,
        })
        # yaw_45 index is used so a rename of TEST_YAW cannot silently score another view.
        if int(angles[str(TEST_YAW)]["per_exercise"]["squat"]["n"]) != int(
            (exercise[test_ids] == 0).sum()
        ):
            raise RuntimeError("45° squat count does not match the test reps")
    seconds = time.perf_counter() - started
    return folds_out, seconds


def _train_all_net(model_name, cache, exercise, mistake, epochs, batch, lr, seed, run_dir):
    seed_everything(seed)
    ids = np.arange(len(exercise))
    poses, ex, y = _stack_views(cache, ids, exercise, mistake, list(range(len(TRAIN_YAWS))), (0, 1))
    loader = DataLoader(
        _RepDataset(poses, ex, y, augment=True, seed=seed),
        batch_size=batch, shuffle=True, num_workers=0,
    )
    model = build_model(model_name)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    started = time.perf_counter()
    for epoch in range(epochs):
        loss = _fit_epoch(model, loader, opt)
        _log(run_dir, f"all-people epoch {epoch + 1}/{epochs} loss {loss:.4f}")
    seconds = time.perf_counter() - started
    return model, seconds


def _train_folds_rf(pack, cache, exercise, mistake, is_correct, seed, run_dir):
    folds_out = []
    started = time.perf_counter()
    pose3d = [rep_pose(pack, i) for i in range(len(exercise))]
    feat3d = np.vstack([feature_vector(features_3d(pose)) for pose in pose3d])
    ref_folds = []
    for fold in loso_folds(pack["subject"]):
        train_ids, test_ids = fold_indices(pack["subject"], fold["test"])
        majority = {
            name: _majority(mistake[train_ids][exercise[train_ids] == index])
            for index, name in enumerate(EXERCISES)
        }
        train_poses, train_ex, train_y = _stack_views(
            cache, train_ids, exercise, mistake, list(range(len(TRAIN_YAWS))), (0, 1),
        )
        forests = _fit_forests(_feature_matrix(train_poses), train_y, train_ex, seed)
        _log(run_dir, f"fold {fold['fold'] + 1} {fold['test']} random forest fit on {len(train_poses)} views")
        angles = {}
        for yaw_i, yaw in enumerate(TRAIN_YAWS):
            view, view_ex, _ = _stack_views(cache, test_ids, exercise, mistake, [yaw_i], (0,))
            pred = _predict_forests(forests, _feature_matrix(view), view_ex)
            angles[str(yaw)] = _score_predictions(
                mistake[test_ids], pred, exercise[test_ids], is_correct[test_ids], majority,
            )
        folds_out.append({
            "fold": int(fold["fold"]),
            "test_subject": fold["test"],
            "n_train_reps": int(len(train_ids)),
            "n_test_reps": int(len(test_ids)),
            "yaw_45": angles[str(TEST_YAW)],
            "angles": angles,
        })
        forests_3d = _fit_forests(feat3d[train_ids], mistake[train_ids], exercise[train_ids], seed)
        pred_3d = _predict_forests(forests_3d, feat3d[test_ids], exercise[test_ids])
        ref_folds.append(_score_predictions(
            mistake[test_ids], pred_3d, exercise[test_ids], is_correct[test_ids], majority,
        ))
    seconds = time.perf_counter() - started
    return folds_out, ref_folds, seconds


def _train_all_rf(cache, exercise, mistake, seed):
    ids = np.arange(len(exercise))
    poses, ex, y = _stack_views(cache, ids, exercise, mistake, list(range(len(TRAIN_YAWS))), (0, 1))
    return _fit_forests(_feature_matrix(poses), y, ex, seed)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def train(model_name, epochs=60, batch=BATCH, lr=1e-3, seed=42, quick=False):
    if model_name not in ("stgcn", "lstm", "rf"):
        raise ValueError(model_name)
    seed_everything(seed)
    if quick:
        epochs = 1
        batch = min(batch, 8)
    pack = load_pack(quick)
    exercise, mistake, is_correct = _labels(pack)
    _log_ready = None
    name = f"m2_{model_name}" + ("_quick" if quick else "")
    run_dir = new_run(name, {
        "model": model_name,
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "seed": seed,
        "quick": quick,
        "num_workers": 0,
        "test_yaw": TEST_YAW,
        "train_yaws": list(TRAIN_YAWS),
        "rf_features": "2d" if model_name == "rf" else None,
        "rf_trees": RF_TREES if model_name == "rf" else None,
        "early_stopping": False,
        "scored_checkpoint": "loso folds only; the all-people checkpoint is not scored",
    })
    _log(run_dir, f"model={model_name} epochs={epochs} quick={quick} reps={len(exercise)}")
    _log_ready = run_dir
    cache, project_s = project_cache(pack)
    _log(run_dir, f"projected {len(exercise)} reps in {project_s:.1f}s")

    reference_3d = None
    if model_name == "rf":
        folds, ref_folds, train_s = _train_folds_rf(pack, cache, exercise, mistake, is_correct, seed, run_dir)
        reference_3d = {
            "note": "3D features, one vector per rep, same people held out. Not the 2D comparison.",
            "folds": [
                {"test_subject": folds[i]["test_subject"], "scores": ref_folds[i]}
                for i in range(len(folds))
            ],
        }
        ref_summary = {"per_exercise": {}}
        for ex_name in EXERCISES:
            mean, std = _mean_std([
                item["scores"]["per_exercise"][ex_name]["accuracy"] for item in reference_3d["folds"]
            ])
            ref_summary["per_exercise"][ex_name] = {"accuracy_mean": mean, "accuracy_std": std}
        reference_3d["summary"] = ref_summary
        all_started = time.perf_counter()
        forests = _train_all_rf(cache, exercise, mistake, seed)
        all_s = time.perf_counter() - all_started
        MODELS.mkdir(parents=True, exist_ok=True)
        ckpt = MODELS / f"{name}.pickle"
        with open(ckpt, "wb") as handle:
            pickle.dump({"model": "rf", "features": "2d", "forests": forests, "exercises": list(EXERCISES)}, handle)
        _log(run_dir, f"all-people checkpoint {ckpt.name} saved and not evaluated")
        parameter_count = None
    else:
        folds, train_s = _train_folds_net(
            model_name, pack, cache, exercise, mistake, is_correct, epochs, batch, lr, seed, run_dir,
        )
        model, all_s = _train_all_net(model_name, cache, exercise, mistake, epochs, batch, lr, seed, run_dir)
        parameter_count = int(sum(p.numel() for p in model.parameters()))
        MODELS.mkdir(parents=True, exist_ok=True)
        ckpt = MODELS / f"{name}.pt"
        torch.save({
            "model": model_name,
            "state_dict": model.state_dict(),
            "exercises": list(EXERCISES),
            "n_mistakes": dict(N_MISTAKES),
        }, ckpt)
        _log(run_dir, f"all-people checkpoint {ckpt.name} saved and not evaluated")

    metrics = {
        "model": model_name,
        "quick": quick,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "num_workers": 0,
        "test_yaw": TEST_YAW,
        "train_views": "yaw 0..90 x mirror, train only",
        "test_views": "one yaw, no mirror, no augmentation",
        "correct_share": _correct_share(exercise, is_correct),
        "project_seconds": round(project_s, 2),
        "train_seconds": round(train_s, 2),
        "all_people_seconds": round(all_s, 2),
        "parameter_count": parameter_count,
        "rf_trees": RF_TREES if model_name == "rf" else None,
        "checkpoint": ckpt.name,
        "checkpoint_scored": False,
        "folds": folds,
        "summary": summarize_folds(folds),
        "reference_3d_rf": reference_3d,
        "context_not_measured": {
            "authors_graph_accuracy": CONTEXT_AUTHORS,
            "earlier_3d_rf_squat_accuracy": CONTEXT_EARLIER_SQUAT_RF,
        },
    }
    save_metrics(run_dir, _jsonable(metrics))
    _log(run_dir, f"train_seconds {train_s:.1f} all_people_seconds {all_s:.1f}")
    _log(run_dir, f"RUN_DIR={run_dir}")
    del _log_ready
    return run_dir


def _fmt_pm(mean, std):
    if mean is None or std is None:
        return "—"
    return f"{100 * mean:.1f} ± {100 * std:.1f}"


def _load_metrics(path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(stgcn_json, lstm_json, rf_json, out_path=None):
    """Write reports/m2_benchmark.md from the three metrics files. No hand-typed scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {
        "stgcn": _load_metrics(stgcn_json),
        "lstm": _load_metrics(lstm_json),
        "rf": _load_metrics(rf_json),
    }
    out_path = REPORTS / "m2_benchmark.md" if out_path is None else out_path
    FIGURES.mkdir(parents=True, exist_ok=True)
    run_dir = new_run("m2_benchmark", {
        "sources": {name: str(path) for name, path in {
            "stgcn": stgcn_json, "lstm": lstm_json, "rf": rf_json,
        }.items()},
    })

    def cell(model, exercise, key):
        return runs[model]["summary"]["per_exercise"][exercise][key]

    lines = [
        "# Model 2 benchmark",
        "",
        "Leave-one-person-out on EC3D, four people. Every model is scored on the same test reps, projected at 45° with no mirror and no augmentation. The random forest uses the six 2D features of that same skeleton.",
        "",
        "The ± value is the sample standard deviation across the four folds (ddof=1). With four people, one person moves the mean a lot.",
        "",
        f"Authors' graph network: {100 * runs['stgcn']['context_not_measured']['authors_graph_accuracy']:.1f}%. Different input and setup, so this is context only, not a target for the table.",
        "",
        f"Earlier 3D random forest on squats: {100 * runs['stgcn']['context_not_measured']['earlier_3d_rf_squat_accuracy']:.0f}%. That number is from the earlier analysis, not from this benchmark.",
        "",
        "## Accuracy at 45° (mean ± std, %)",
        "",
        "| Model | Squat | Lunge | Plank | Train minutes |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in ("stgcn", "lstm", "rf"):
        minutes = runs[model]["train_seconds"] / 60
        lines.append("| {model} | {sq} | {lu} | {pl} | {mins:.2f} |".format(
            model=model,
            sq=_fmt_pm(cell(model, "squat", "accuracy_mean"), cell(model, "squat", "accuracy_std")),
            lu=_fmt_pm(cell(model, "lunge", "accuracy_mean"), cell(model, "lunge", "accuracy_std")),
            pl=_fmt_pm(cell(model, "plank", "accuracy_mean"), cell(model, "plank", "accuracy_std")),
            mins=minutes,
        ))
    lines += ["", "## Macro-F1 at 45° (mean ± std, %)", "", "| Model | Squat | Lunge | Plank |", "|---|---:|---:|---:|"]
    for model in ("stgcn", "lstm", "rf"):
        lines.append("| {model} | {sq} | {lu} | {pl} |".format(
            model=model,
            sq=_fmt_pm(cell(model, "squat", "macro_f1_mean"), cell(model, "squat", "macro_f1_std")),
            lu=_fmt_pm(cell(model, "lunge", "macro_f1_mean"), cell(model, "lunge", "macro_f1_std")),
            pl=_fmt_pm(cell(model, "plank", "macro_f1_mean"), cell(model, "plank", "macro_f1_std")),
        ))
    lines += [
        "",
        "## Mistake detection at 45°",
        "",
        "Positive class = not correct. Specificity is the share of correct reps recognized as correct. A model that always says \"mistake\" has recall 100% and specificity 0%. Majority accuracy predicts the most common training class of that exercise.",
        "",
        "| Model | Exercise | Recall | Precision | Specificity | Majority accuracy | Always-mistake precision |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("stgcn", "lstm", "rf"):
        for exercise in EXERCISES:
            lines.append("| {model} | {exercise} | {rec} | {prec} | {spec} | {maj} | {always_p} |".format(
                model=model,
                exercise=exercise,
                rec=_fmt_pm(cell(model, exercise, "mistake_recall_mean"), cell(model, exercise, "mistake_recall_std")),
                prec=_fmt_pm(cell(model, exercise, "mistake_precision_mean"), cell(model, exercise, "mistake_precision_std")),
                spec=_fmt_pm(cell(model, exercise, "mistake_specificity_mean"), cell(model, exercise, "mistake_specificity_std")),
                maj=_fmt_pm(cell(model, exercise, "majority_accuracy_mean"), cell(model, exercise, "majority_accuracy_std")),
                always_p=_fmt_pm(
                    cell(model, exercise, "always_mistake_precision_mean"),
                    cell(model, exercise, "always_mistake_precision_std"),
                ),
            ))
    share = runs["stgcn"]["correct_share"]
    lines += [
        "",
        "Correct share in this run's rep table: "
        + ", ".join(f"{name} {100 * share[name]:.1f}%" for name in EXERCISES)
        + ".",
        "",
        "## Per fold, accuracy at 45° (%)",
        "",
    ]
    subjects = [fold["test_subject"] for fold in runs["stgcn"]["folds"]]
    header = "| Model | Exercise | " + " | ".join(subjects) + " |"
    lines += [header, "|---|---|" + "---:|" * len(subjects)]
    for model in ("stgcn", "lstm", "rf"):
        for exercise in EXERCISES:
            cells = []
            for fold in runs[model]["folds"]:
                value = fold["yaw_45"]["per_exercise"][exercise]["accuracy"]
                cells.append("—" if value is None else f"{100 * value:.1f}")
            lines.append(f"| {model} | {exercise} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Accuracy by camera angle (mean ± std, %)",
        "",
        "Same test reps, no mirror. 45° is the main result. The other angles are the robustness check.",
        "",
        "| Model | Exercise | " + " | ".join(str(yaw) + "°" for yaw in TRAIN_YAWS) + " |",
        "|---|---|" + "---:|" * len(TRAIN_YAWS),
    ]
    for model in ("stgcn", "lstm", "rf"):
        for exercise in list(EXERCISES) + ["overall"]:
            cells = []
            for yaw in TRAIN_YAWS:
                block = runs[model]["summary"]["angles"][str(yaw)][exercise]
                cells.append(_fmt_pm(block["accuracy_mean"], block["accuracy_std"]))
            lines.append(f"| {model} | {exercise} | " + " | ".join(cells) + " |")

    ref = runs["rf"]["reference_3d_rf"]
    lines += [
        "",
        "## Random forest on 3D features (separate)",
        "",
        ref["note"],
        "",
        "| Exercise | Accuracy (mean ± std, %) |",
        "|---|---:|",
    ]
    for exercise in EXERCISES:
        block = ref["summary"]["per_exercise"][exercise]
        lines.append(f"| {exercise} | {_fmt_pm(block['accuracy_mean'], block['accuracy_std'])} |")
    lines += [
        "",
        "## Checkpoints",
        "",
        "Each run also saves a checkpoint trained on all four people (`models/`). That checkpoint is for live use. It is not scored, because it has seen every person.",
        "",
        "| Model | LOSO train minutes | All-people minutes | Parameters | Checkpoint | Source |",
        "|---|---:|---:|---:|---|---|",
    ]
    for model, path in (("stgcn", stgcn_json), ("lstm", lstm_json), ("rf", rf_json)):
        info = runs[model]
        params = "—" if info["parameter_count"] is None else str(info["parameter_count"])
        lines.append(
            f"| {model} | {info['train_seconds'] / 60:.2f} | {info['all_people_seconds'] / 60:.2f} | {params} | `{info['checkpoint']}` | `{path}` |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(EXERCISES))
    width = 0.25
    for i, model in enumerate(("stgcn", "lstm", "rf")):
        means = [cell(model, name, "accuracy_mean") for name in EXERCISES]
        stds = [cell(model, name, "accuracy_std") for name in EXERCISES]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, label=model, capsize=3)
    ax.set_xticks(x, list(EXERCISES))
    ax.set_ylabel("Accuracy at 45°")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Model 2 accuracy per exercise")
    fig.tight_layout()
    bar_path = run_dir / "accuracy_per_exercise.png"
    fig.savefig(bar_path, dpi=120)
    fig.savefig(FIGURES / "m2_accuracy_per_exercise.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for model in ("stgcn", "lstm", "rf"):
        ys = [runs[model]["summary"]["angles"][str(yaw)]["overall"]["accuracy_mean"] for yaw in TRAIN_YAWS]
        ax.plot(list(TRAIN_YAWS), ys, marker="o", label=model)
    ax.set_xlabel("Camera angle (degrees)")
    ax.set_ylabel("Overall accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Model 2 accuracy by camera angle")
    fig.tight_layout()
    fig.savefig(run_dir / "accuracy_by_angle.png", dpi=120)
    fig.savefig(FIGURES / "m2_accuracy_by_angle.png", dpi=120)
    plt.close(fig)

    save_metrics(run_dir, {
        "sources": {model: str(path) for model, path in {
            "stgcn": stgcn_json, "lstm": lstm_json, "rf": rf_json,
        }.items()},
        "report": str(out_path),
        "bar_chart": str(bar_path),
    })
    print(f"wrote {out_path}", flush=True)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train Model 2")
    parser.add_argument("--model", choices=["stgcn", "lstm", "rf"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--report", nargs=3, metavar="METRICS_JSON",
                        help="stgcn, lstm and rf metrics.json paths")
    args = parser.parse_args(argv)
    if args.report:
        write_report(*[__import__("pathlib").Path(p) for p in args.report])
        return
    if not args.model:
        parser.error("--model is required unless --report is set")
    train(args.model, args.epochs, args.batch, args.lr, args.seed, args.quick)


if __name__ == "__main__":
    main()
