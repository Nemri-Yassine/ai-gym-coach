"""Collect experiments/*/metrics.json into reports/results.md."""
from __future__ import annotations

import json
from pathlib import Path

from awc.config import EXPERIMENTS, REPORTS


def _load_all():
    rows = []
    for path in sorted(EXPERIMENTS.glob("*/metrics.json")):
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append({"run": path.parent.name, "path": str(path), "metrics": metrics})
    return rows


def write_results(out: Path | None = None):
    out = out or (REPORTS / "results.md")
    rows = _load_all()
    by = {r["run"]: r["metrics"] for r in rows}
    lines = [
        "# Results · RQ1–RQ4 + Big Data",
        "",
        "Numbers below are copied from `experiments/*/metrics.json` (and the week reports).",
        "",
        "## RQ1 · Recognition on unseen people",
        "",
        "- Best Model 1: transformer pose+IMU, macro-F1 **0.880** on `test_unseen` (`reports/m1_benchmark.md`).",
        "- Rep counting macro MAE **0.49** on 616 MM-Fit sets (`reports/rep_counting.md`).",
        "",
        "## RQ2 · Heart rate vs movement (B4 vs B3)",
        "",
    ]
    b3 = by.get("20261107_m3_b3", {})
    b4 = by.get("20261107_m3_b4", {})
    b2 = by.get("20261107_m3_b2", {})
    b1 = by.get("20261107_m3_b1", {})
    lines += [
        f"- B4 (personal+HR) risk AUC **{b4.get('risk_auc')}**, lead **{b4.get('lead_time')}**, FA **{b4.get('false_alarms')}**",
        f"- B3 (personal, no HR) risk AUC **{b3.get('risk_auc')}**, lead **{b3.get('lead_time')}**, FA **{b3.get('false_alarms')}**",
        "- Honest finding: HR does **not** improve risk AUC on this MM-Fit setup. See `reports/m3_results.md`.",
        "",
        "## RQ3 · Personal vs general (B4 vs B2) and vs rule (B1)",
        "",
        f"- B2 (general+HR) AUC **{b2.get('risk_auc')}** — better than B4 here.",
        f"- B1 rule AUC **{b1.get('risk_auc')}**, lead **{b1.get('lead_time')}**.",
        "- Within-session personal baseline does not beat general normalisation on these labels.",
        "",
        "## RQ4 · Raw MM-Fit video",
        "",
        "- w20 MediaPipe vs dataset pose: M1 pose-only 0.85→0.80, rep MAE 0.93→1.56 (`reports/rq4_first_look.md`).",
        "",
        "## Big data",
        "",
        "- Spark sync scaling: see week1 / `experiments/*_spark_scaling` if present.",
        "- Kafka replay: pose 29.8 msg/s @ 48 ms; HR 0.79 msg/s @ 19 ms (`experiments/20261107_kafka_replay`).",
        "- Rep table: 5,918 rows (`data/processed/rep_table_summary.json`).",
        "",
        "## Runs indexed",
        "",
    ]
    for r in rows:
        lines.append(f"- `{r['run']}`")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main():
    path = write_results()
    print("wrote", path)


if __name__ == "__main__":
    main()
