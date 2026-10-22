"""Train the 9 Model 1 runs and write reports/m1_benchmark.md.

    python scripts/run_m1_benchmark.py
    python scripts/run_m1_benchmark.py --quick
"""
import argparse
import json
from pathlib import Path

from awc.config import EXPERIMENTS, REPORTS
from awc.training.train_m1 import train

ARCHS = ("cnn", "lstm", "transformer")
INPUTS = ("pose", "imu", "both")


def latest_metrics(arch, inputs, quick):
    name = f"m1_{arch}_{inputs}" + ("_quick" if quick else "")
    folders = sorted(EXPERIMENTS.glob(f"*_{name}"))
    if not folders:
        raise FileNotFoundError(name)
    return json.loads((folders[-1] / "metrics.json").read_text(encoding="utf-8")), folders[-1]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)
    rows = []
    for arch in ARCHS:
        for inputs in INPUTS:
            print(f"=== {arch} {inputs} ===", flush=True)
            train(arch=arch, inputs=inputs, quick=args.quick)
            metrics, folder = latest_metrics(arch, inputs, args.quick)
            rows.append((arch, inputs, metrics, folder))
    lines = [
        "# Model 1 benchmark",
        "",
        "Test set: MM-Fit `test_unseen` (people not in training).",
        "",
        "| Architecture | Inputs | Macro-F1 unseen | Accuracy unseen | Macro-F1 seen | Params | Train minutes |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    best = None
    for arch, inputs, metrics, folder in rows:
        f1 = metrics.get("macro_f1_unseen")
        acc = metrics.get("accuracy_unseen")
        seen = metrics.get("macro_f1_seen")
        minutes = None if metrics.get("train_seconds") is None else round(metrics["train_seconds"] / 60, 2)
        lines.append(f"| {arch} | {inputs} | {f1} | {acc} | {seen} | {metrics.get('parameter_count')} | {minutes} |")
        if f1 is not None and (best is None or f1 > best[0]):
            best = (f1, arch, inputs, folder)
    if best:
        lines += ["", f"Best unseen macro-F1: **{best[0]}** with `{best[1]}` / `{best[2]}` (`{best[3].name}`)."]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "m1_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORTS / 'm1_benchmark.md'}")


if __name__ == "__main__":
    main()
