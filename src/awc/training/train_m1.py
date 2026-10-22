"""Train Model 1. One run = one architecture and one input set.

    python -m awc.training.train_m1 --arch cnn --inputs both
    python -m awc.training.train_m1 --arch cnn --inputs both --quick
"""
import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from awc.config import MMFIT_CLASSES, MODELS, PROCESSED
from awc.evaluation.metrics import accuracy, confusion, macro_f1, per_class_f1, plot_confusion
from awc.models.m1_recognition import M1Model
from awc.training.tracking import new_run, save_metrics, seed_everything

QUICK_TRAIN = 512


def class_weights(y, n_classes=11):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


def _synthetic(n=600, seed=42):
    rng = np.random.default_rng(seed)
    split = np.array(["train"] * 512 + ["val"] * 32 + ["test_seen"] * 28 + ["test_unseen"] * 28)
    return {
        "X_pose": rng.normal(size=(n, 90, 36)).astype(np.float32),
        "X_imu": rng.normal(size=(n, 90, 6)).astype(np.float32),
        "imu_mask": np.ones(n, np.float32),
        "y": rng.integers(0, 11, size=n),
        "split": split,
    }


def load_windows(quick):
    path = PROCESSED / "m1_windows.npz"
    if not path.exists():
        if quick:
            return _synthetic()
        raise FileNotFoundError(f"{path} is missing. Build it with python -m awc.features.windows")
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _loader(pack, split, batch, shuffle):
    m = pack["split"] == split
    if split == "train" and len(pack["y"]) > QUICK_TRAIN and pack.get("_quick_cap"):
        idx = np.flatnonzero(m)[:QUICK_TRAIN]
    else:
        idx = np.flatnonzero(m)
    ds = TensorDataset(
        torch.from_numpy(np.asarray(pack["X_pose"][idx], dtype=np.float32)),
        torch.from_numpy(np.asarray(pack["X_imu"][idx], dtype=np.float32)),
        torch.from_numpy(np.asarray(pack["imu_mask"][idx], dtype=np.float32)),
        torch.from_numpy(np.asarray(pack["y"][idx], dtype=np.int64)),
    )
    drop = shuffle and len(ds) > batch
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=drop)


def _predict(model, loader, inputs):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for pose, imu, mask, y in loader:
            if inputs == "pose":
                logits = model(pose)
            elif inputs == "imu":
                logits = model(imu)
            else:
                logits = model(pose, imu, mask)
            ys.append(y.numpy())
            ps.append(logits.argmax(dim=1).numpy())
    if not ys:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(ys), np.concatenate(ps)


def train(arch, inputs, epochs=30, batch=128, lr=1e-3, seed=42, quick=False):
    seed_everything(seed)
    if quick:
        epochs = 1
        batch = min(batch, 128)
    pack = load_windows(quick)
    if quick:
        pack = dict(pack)
        pack["_quick_cap"] = True
        # Keep evaluation small when the full file is loaded.
        for split, cap in (("val", 256), ("test_seen", 256), ("test_unseen", 256)):
            idx = np.flatnonzero(pack["split"] == split)
            if len(idx) > cap:
                drop = idx[cap:]
                pack["split"] = np.array(pack["split"], copy=True)
                pack["split"][drop] = "unused"
    name = f"m1_{arch}_{inputs}" + ("_quick" if quick else "")
    run_dir = new_run(name, {
        "model": "m1", "arch": arch, "inputs": inputs, "epochs": epochs,
        "batch": batch, "lr": lr, "seed": seed, "quick": quick,
    })
    (run_dir / "log.txt").write_text(f"arch={arch} inputs={inputs} epochs={epochs} quick={quick}\n", encoding="utf-8")

    model = M1Model(arch, inputs)
    train_idx = np.flatnonzero(pack["split"] == "train")
    y_for_weights = pack["y"][train_idx][:QUICK_TRAIN] if quick else pack["y"][train_idx]
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_for_weights))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = _loader(pack, "train", batch, shuffle=True)
    val_loader = _loader(pack, "val", batch, shuffle=False)

    best_f1 = -1.0
    best_state = None
    stale = 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total, seen = 0.0, 0
        for pose, imu, mask, y in train_loader:
            opt.zero_grad()
            if inputs == "pose":
                logits = model(pose)
            elif inputs == "imu":
                logits = model(imu)
            else:
                logits = model(pose, imu, mask)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(y)
            seen += len(y)
        yv, pv = _predict(model, val_loader, inputs)
        val_f1 = macro_f1(yv, pv) if len(yv) else 0.0
        line = f"epoch {epoch + 1} train_loss {total / max(seen, 1):.4f} val_macro_f1 {val_f1:.4f}\n"
        with (run_dir / "log.txt").open("a", encoding="utf-8") as f:
            f.write(line)
        print(line, end="")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 5:
                print("early stop")
                break
    train_s = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    labels = list(range(len(MMFIT_CLASSES)))
    metrics = {
        "arch": arch,
        "inputs": inputs,
        "quick": quick,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "train_seconds": round(train_s, 2),
        "best_val_macro_f1": None if best_f1 < 0 else round(best_f1, 4),
        "epochs_ran": epoch + 1,
    }
    for split in ("test_unseen", "test_seen"):
        yt, pt = _predict(model, _loader(pack, split, batch, shuffle=False), inputs)
        key = "unseen" if split == "test_unseen" else "seen"
        if len(yt) == 0:
            metrics[f"accuracy_{key}"] = None
            metrics[f"macro_f1_{key}"] = None
            metrics[f"per_class_f1_{key}"] = {}
            continue
        metrics[f"accuracy_{key}"] = round(accuracy(yt, pt), 4)
        metrics[f"macro_f1_{key}"] = round(macro_f1(yt, pt), 4)
        metrics[f"per_class_f1_{key}"] = {
            MMFIT_CLASSES[int(k)]: round(v, 4)
            for k, v in per_class_f1(yt, pt, labels).items()
        }
        if split == "test_unseen":
            cm = confusion(yt, pt, labels)
            plot_confusion(cm, MMFIT_CLASSES, run_dir / "confusion_test_unseen.png",
                           title=f"M1 {arch} {inputs} test_unseen")
            metrics["confusion_test_unseen"] = cm.tolist()

    MODELS.mkdir(parents=True, exist_ok=True)
    ckpt = MODELS / f"{name}.pt"
    torch.save({"arch": arch, "inputs": inputs, "state_dict": model.state_dict(), "classes": MMFIT_CLASSES}, ckpt)
    metrics["checkpoint"] = ckpt.name
    save_metrics(run_dir, metrics)
    print(f"RUN_DIR={run_dir}")
    return run_dir


def main(argv=None):
    p = argparse.ArgumentParser(description="Train Model 1")
    p.add_argument("--arch", choices=["cnn", "lstm", "transformer"], required=True)
    p.add_argument("--inputs", choices=["pose", "imu", "both"], required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)
    train(args.arch, args.inputs, args.epochs, args.batch, args.lr, args.seed, args.quick)


if __name__ == "__main__":
    main()
