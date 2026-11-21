"""Model 2 v2: absolute vs relative (within-person) judgement, from saved out-of-fold predictions.

Absolute: is this set bad?  (mean P(correct) over its windows < 0.5)
Relative: of this person's two sets (same exercise, same camera view), which one is worse?
          Correct when the bad set gets the lower mean P(correct). Both sets are separate
          recordings, so "same recording as the reference" cannot leak (unlike calibration).
Control:  the same with the body-proportion model (camera / body shape only).

Run after `python -m awc.training.train_m2_gym` (writes oof.npz):
    python scripts/analysis/m2_relative.py [run_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def set_table(z, preds):
    """One row per set (video): person, exercise, view, label, mean P per model."""
    ev = z["evaluate"]
    df = pd.DataFrame({k: z[k][ev] for k in ("subject", "exercise", "view", "video", "y")})
    for name, p in preds.items():
        df[name] = p[ev]
    return df.groupby(["subject", "exercise", "view", "video"], as_index=False).mean(numeric_only=True)


def relative_accuracy(sets, model):
    """Fraction of (person, exercise, view) pairs where the bad set scores lower. Ties count half."""
    out = []
    for (_s, ex, _v), g in sets.groupby(["subject", "exercise", "view"]):
        if set(g.y) != {0.0, 1.0}:
            continue
        good, bad = g[g.y == 1][model].iloc[0], g[g.y == 0][model].iloc[0]
        out.append((ex, 1.0 if good > bad else 0.5 if good == bad else 0.0, good - bad))
    return pd.DataFrame(out, columns=["exercise", "hit", "margin"])


def main(run_dir=None):
    run_dir = Path(run_dir) if run_dir else sorted((ROOT / "experiments").glob("*_m2_gym_windows"))[-1]
    z = np.load(run_dir / "oof.npz", allow_pickle=False)
    preds = {k[2:]: z[k] for k in z.files if k.startswith("p_")}
    if (run_dir / "oof_tcn.npy").exists():
        preds["tcn"] = np.load(run_dir / "oof_tcn.npy")
    plain = [k for k in ("rf", "hgb", "tcn") if k in preds]
    preds["ensemble"] = np.mean([preds[k] for k in plain], axis=0)
    sets = set_table(z, preds)

    rows, per_ex = [], {}
    for model in [*plain, "ensemble", "prop"]:
        absolute = float((((sets[model] >= 0.5).astype(float)) == sets.y).mean())
        rel = relative_accuracy(sets, model)
        rows.append({"model": model, "absolute_set_acc": round(absolute, 3),
                     "relative_pair_acc": round(float(rel.hit.mean()), 3), "pairs": int(len(rel)),
                     "median_margin": round(float(rel.margin.median()), 3)})
        per_ex[model] = rel.groupby("exercise").hit.mean().round(3).to_dict()
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print("\nrelative accuracy per exercise:")
    print(pd.DataFrame(per_ex).to_string())
    (run_dir / "relative.json").write_text(json.dumps({"summary": rows, "per_exercise": per_ex}, indent=2))
    print(f"\nwrote {run_dir / 'relative.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
