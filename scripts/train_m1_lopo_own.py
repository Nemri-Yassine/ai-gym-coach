"""Leave-one-person-out adaptation of pose-only transformer on own sets.

    python scripts/train_m1_lopo_own.py

Fine-tunes m1_transformer_pose on MM-Fit train windows + one person's own eval
sets; tests on the other person. Both directions. No threshold tuning on the
test person. Saves models/m1_transformer_pose_lopo_<test_person>.pt — never
overwrites m1_transformer_pose.pt. Reports next to zero-shot in rq4_own.md.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from awc.config import (
    FIGURES,
    MMFIT_CLASSES,
    MODELS,
    OWN_INTERIM,
    PROCESSED,
    REPORTS,
    models_dir,
)
from awc.data.own_manifest import load_manifest
from awc.evaluation.metrics import macro_f1
from awc.features.skeleton import normalize
from awc.models.m1_recognition import M1Model
from awc.training.tracking import new_run, save_metrics, seed_everything

WINDOW, HOP = 90, 45
POSE_X = [f"x{i}" for i in range(18)]
POSE_Y = [f"y{i}" for i in range(18)]
EPOCHS = 8
LR = 1e-4
BATCH = 64
SEED = 42
PRIMARY = "m1_transformer_pose"


def _windows_from_pose(pose: np.ndarray, y_idx: int):
    out_x, out_y = [], []
    for s in range(0, len(pose) - WINDOW + 1, HOP):
        w = pose[s:s + WINDOW]
        if np.isfinite(w).all():
            out_x.append(normalize(w).reshape(WINDOW, 36))
            out_y.append(y_idx)
    if not out_x:
        return np.zeros((0, WINDOW, 36), np.float32), np.zeros(0, np.int64)
    return np.asarray(out_x, np.float32), np.asarray(out_y, np.int64)


def own_windows(person: str):
    xs, ys, keys = [], [], []
    for r in load_manifest():
        if r["use"] != "eval" or r["person"] != person:
            continue
        path = OWN_INTERIM / person / f"{r['session_folder']}_set{int(r['set']):02d}.parquet"
        if not path.is_file():
            continue
        df = pd.read_parquet(path)
        pose = np.stack([df[POSE_X].to_numpy(), df[POSE_Y].to_numpy()], axis=-1)
        good = np.isfinite(pose).all(axis=(1, 2))
        if good.sum() < WINDOW:
            continue
        x, y = _windows_from_pose(pose[good], MMFIT_CLASSES.index(r["exercise"]))
        if len(x):
            xs.append(x)
            ys.append(y)
            keys.append(f"{person}/{r['session_folder']}/set{int(r['set']):02d}")
    if not xs:
        return np.zeros((0, WINDOW, 36), np.float32), np.zeros(0, np.int64), []
    return np.concatenate(xs), np.concatenate(ys), keys


def mmfit_train_pose_windows(cap: int | None = None):
    pack = np.load(PROCESSED / "m1_windows.npz", allow_pickle=True)
    pose = pack["X_pose"]
    y = pack["y"]
    split = pack["split"]
    m = split == "train"
    x = pose[m].astype(np.float32)
    yy = y[m].astype(np.int64)
    if cap and len(x) > cap:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(x), size=cap, replace=False)
        x, yy = x[idx], yy[idx]
    return x, yy


def _vote(preds):
    na = MMFIT_CLASSES.index("non_activity")
    active = [int(p) for p in preds if p != na]
    if not active:
        return "no_exercise_detected"
    return MMFIT_CLASSES[Counter(active).most_common(1)[0][0]]


def eval_person(model, person: str):
    rows = [r for r in load_manifest() if r["use"] == "eval" and r["person"] == person]
    correct = 0
    preds = []
    for r in rows:
        path = OWN_INTERIM / person / f"{r['session_folder']}_set{int(r['set']):02d}.parquet"
        if not path.is_file():
            preds.append({"key": f"{person}/{r['session_folder']}/set{int(r['set']):02d}",
                          "truth": r["exercise"], "pred": "missing", "ok": False})
            continue
        df = pd.read_parquet(path)
        pose = np.stack([df[POSE_X].to_numpy(), df[POSE_Y].to_numpy()], axis=-1)
        good = np.isfinite(pose).all(axis=(1, 2))
        x, _ = _windows_from_pose(pose[good], 0)
        if len(x) == 0:
            pred = "no_exercise_detected"
        else:
            with torch.no_grad():
                logits = model(torch.from_numpy(x))
                pred = _vote(logits.argmax(1).numpy())
        ok = pred == r["exercise"]
        correct += int(ok)
        preds.append({"key": f"{person}/{r['session_folder']}/set{int(r['set']):02d}",
                      "truth": r["exercise"], "pred": pred, "ok": ok})
    n = len(rows)
    return {
        "n_sets": n,
        "set_accuracy": round(correct / n, 3) if n else None,
        "set_correct": correct,
        "preds": preds,
    }


def finetune(adapt_person: str, test_person: str):
    seed_everything(SEED)
    base = MODELS / f"{PRIMARY}.pt"
    ckpt = torch.load(base, map_location="cpu", weights_only=False)
    model = M1Model(ckpt["arch"], ckpt["inputs"])
    model.load_state_dict(ckpt["state_dict"])

    x_mm, y_mm = mmfit_train_pose_windows(cap=8000)
    x_own, y_own, own_keys = own_windows(adapt_person)
    x = np.concatenate([x_mm, x_own]) if len(x_own) else x_mm
    y = np.concatenate([y_mm, y_own]) if len(y_own) else y_mm

    # class weights on combined
    counts = np.bincount(y, minlength=len(MMFIT_CLASSES)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = (counts.sum() / (len(counts) * counts)).astype(np.float32)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(w))
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    run = new_run(
        f"m1_lopo_adapt_{adapt_person}_test_{test_person}",
        {
            "adapt_person": adapt_person,
            "test_person": test_person,
            "epochs": EPOCHS,
            "lr": LR,
            "n_mmfit": int(len(x_mm)),
            "n_own": int(len(x_own)),
            "own_sets": own_keys,
            "seed": SEED,
        },
    )

    model.train()
    n = len(x)
    for epoch in range(EPOCHS):
        perm = np.random.default_rng(SEED + epoch).permutation(n)
        total, seen = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            pose = torch.from_numpy(x[idx])
            yy = torch.from_numpy(y[idx])
            opt.zero_grad()
            loss = loss_fn(model(pose), yy)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yy)
            seen += len(yy)
        print(f"{adapt_person}→{test_person} epoch {epoch + 1} loss {total / max(seen, 1):.4f}")

    model.eval()
    zero_shot = eval_person(
        _load_base(), test_person
    )
    adapted = eval_person(model, test_person)

    out = models_dir()
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / f"m1_transformer_pose_lopo_{test_person}.pt"
    torch.save({
        "arch": ckpt["arch"], "inputs": "pose",
        "state_dict": model.state_dict(), "classes": MMFIT_CLASSES,
        "adapt_person": adapt_person, "test_person": test_person,
        "lopo": True,
    }, ckpt_path)

    metrics = {
        "adapt_person": adapt_person,
        "test_person": test_person,
        "n_own_adapt_windows": int(len(x_own)),
        "n_own_adapt_sets": len(own_keys),
        "zero_shot": zero_shot,
        "adapted": adapted,
        "checkpoint": ckpt_path.name,
    }
    save_metrics(run, metrics)
    print(f"test {test_person}: zero-shot {zero_shot['set_accuracy']} → LOPO {adapted['set_accuracy']} "
          f"(n={adapted['n_sets']}) ckpt={ckpt_path.name}")
    return metrics


def _load_base():
    ckpt = torch.load(MODELS / f"{PRIMARY}.pt", map_location="cpu", weights_only=False)
    model = M1Model(ckpt["arch"], ckpt["inputs"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def main():
    results = [
        finetune("yassine", "partner"),
        finetune("partner", "yassine"),
    ]
    summary = {
        "directions": results,
        "note": "LOPO own adaptation; zero-shot remains the main RQ4 result",
    }
    run = new_run("m1_lopo_own_summary", {"seed": SEED})
    save_metrics(run, summary)
    # Append a short block to rq4_own.md if present
    path = REPORTS / "rq4_own.md"
    block = [
        "",
        "## LOPO own adaptation (separate experiment)",
        "",
        "Fine-tune pose-only transformer on MM-Fit train + one person's own eval sets; "
        "test on the other person. No threshold tuning on the test person. "
        "**Zero-shot remains the main RQ4 number.**",
        "",
        "| Adapt on | Test on | n test sets | Zero-shot set acc | LOPO set acc | Checkpoint |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in results:
        block.append(
            f"| {r['adapt_person']} | {r['test_person']} | {r['adapted']['n_sets']} | "
            f"{r['zero_shot']['set_accuracy']} | **{r['adapted']['set_accuracy']}** | "
            f"`{r['checkpoint']}` |"
        )
    block += [
        "",
        f"n = 8 (partner) / 12 (yassine) eval sets. Metrics: `{run.as_posix()}/metrics.json`.",
        "",
    ]
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        marker = "## LOPO own adaptation"
        if marker in text:
            text = text.split(marker)[0].rstrip() + "\n"
        path.write_text(text.rstrip() + "\n" + "\n".join(block), encoding="utf-8")
        print(f"updated {path}")
    print(f"SUMMARY_DIR={run}")


if __name__ == "__main__":
    main()
