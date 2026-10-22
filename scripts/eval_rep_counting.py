"""Count reps in every labelled MM-Fit set and write the per-exercise table.

    python scripts/eval_rep_counting.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from awc.config import FIGURES, REPORTS
from awc.data import mmfit
from awc.evaluation.metrics import rep_count_errors
from awc.features.reps import count_reps
from awc.features.skeleton import normalize
from awc.features.sync import POSE_X, POSE_Y, pose_to_frame
from awc.training.tracking import new_run, save_metrics


def set_pose(pose_df, start, end):
    sub = pose_df[(pose_df["frame"] >= start) & (pose_df["frame"] <= end)]
    if len(sub) < 15:
        return None
    seq = np.stack([
        sub[POSE_X].to_numpy(dtype=np.float64),
        sub[POSE_Y].to_numpy(dtype=np.float64),
    ], axis=-1)
    return normalize(seq)


def evaluate():
    rows = []
    for session in mmfit.sessions():
        pose = pose_to_frame(mmfit.load_stream(session, "pose_2d", mmap=False))
        for start, end, reps, exercise in mmfit.load_labels(session):
            seq = set_pose(pose, start, end)
            if seq is None:
                continue
            pred, _ = count_reps(seq)
            rows.append({"session": session, "exercise": exercise, "true": int(reps), "pred": int(pred)})
    by_ex = {}
    for row in rows:
        by_ex.setdefault(row["exercise"], {"true": [], "pred": []})
        by_ex[row["exercise"]]["true"].append(row["true"])
        by_ex[row["exercise"]]["pred"].append(row["pred"])
    table = {ex: rep_count_errors(v["true"], v["pred"]) for ex, v in sorted(by_ex.items())}
    maes = [v["mae"] for v in table.values() if v["mae"] is not None]
    summary = {
        "per_exercise": table,
        "macro_mae": float(np.mean(maes)) if maes else None,
        "micro": rep_count_errors([r["true"] for r in rows], [r["pred"] for r in rows]),
        "n_sets": len(rows),
    }
    return summary


def write_report(summary, run_dir):
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Rep counting",
        "",
        f"Sets evaluated: {summary['n_sets']}. Macro MAE (mean of per-exercise MAEs): {summary['macro_mae']:.3f}.",
        f"Micro MAE (every set equal): {summary['micro']['mae']:.3f}.",
        "",
        "| Exercise | Sets | Exact | Within ±1 | MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    names, maes = [], []
    for ex, stats in summary["per_exercise"].items():
        lines.append(
            f"| {ex} | {stats['n']} | {stats['exact'] * 100:.1f}% | {stats['within_1'] * 100:.1f}% | {stats['mae']:.3f} |"
        )
        names.append(ex)
        maes.append(stats["mae"])
    text = "\n".join(lines) + "\n"
    (REPORTS / "rep_counting.md").write_text(text, encoding="utf-8")
    (run_dir / "rep_counting.md").write_text(text, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(names)), maes, color="#1E4BD2")
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
    ax.set_ylabel("MAE (reps)")
    ax.set_title("Rep counting error per exercise")
    fig.tight_layout()
    fig.savefig(FIGURES / "rep_counting_mae.png", dpi=120)
    fig.savefig(run_dir / "rep_counting_mae.png", dpi=120)
    plt.close(fig)


def main():
    summary = evaluate()
    run_dir = new_run("rep_counting", {"fps": 30, "signal": "pca+smooth9", "prominence": 0.3, "min_seconds": 0.6})
    save_metrics(run_dir, summary)
    write_report(summary, run_dir)
    print(json.dumps({"macro_mae": summary["macro_mae"], "micro_mae": summary["micro"]["mae"], "n_sets": summary["n_sets"]}, indent=2))
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
