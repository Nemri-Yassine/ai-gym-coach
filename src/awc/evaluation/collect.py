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


def _auc(m):
    if not m:
        return None
    if "val_tuned" in m and isinstance(m["val_tuned"], dict):
        return m["val_tuned"].get("risk_auc")
    return m.get("risk_auc")


def write_results(out: Path | None = None):
    out = out or (REPORTS / "results.md")
    rows = _load_all()
    by = {r["run"]: r["metrics"] for r in rows}
    # Prefer post-3.2b multi-seed runs
    b3 = by.get("20261107_m3_b3_v2") or by.get("20261107_m3_b3", {})
    b4 = by.get("20261107_m3_b4_v2") or by.get("20261107_m3_b4", {})
    b2 = by.get("20261107_m3_b2_v2") or by.get("20261107_m3_b2", {})
    b1 = by.get("20261107_m3_b1_v2") or by.get("20261107_m3_b1", {})
    b5 = by.get("20261107_m3_b5_v2") or by.get("20261107_m3_b5", {})
    coach_k = by.get("20261107_coach_kafka", {})
    rq4 = by.get("20261107_rq4_mmfit_video", {})
    lines = [
        "# Results · RQ1–RQ4 + Big Data",
        "",
        "Numbers below are copied from `experiments/*/metrics.json` (and the week reports).",
        "Model 3 numbers prefer `*_v2` (Step 3.2b: fair B1, speed-only labels, 3 seeds).",
        "",
        "## RQ1 · Recognition on unseen people",
        "",
        "- Best Model 1: transformer pose+IMU, macro-F1 **0.880** on `test_unseen` (`reports/m1_benchmark.md`).",
        "- Rep counting macro MAE **0.49** on 616 MM-Fit sets (`reports/rep_counting.md`).",
        "",
        "## RQ2 · Heart rate vs movement (B4 vs B3)",
        "",
        f"- B4 (personal+HR) risk AUC **{_auc(b4)}**",
        f"- B3 (personal, no HR) risk AUC **{_auc(b3)}**",
        "- Honest finding: HR does **not** improve risk AUC on this MM-Fit setup. See `reports/m3_results.md`.",
        "",
        "## RQ3 · Personal vs general (B4 vs B2) and vs rule (B1)",
        "",
        f"- B2 (general+HR) AUC **{_auc(b2)}** — better than B4 here.",
        f"- B5 (general, no HR) AUC **{_auc(b5)}**.",
        f"- Fair B1 continuous AUC **{_auc(b1)}** (alert @25% / @20% in `reports/m3_results.md`).",
        "- Within-session personal baseline does not beat general normalisation on these labels.",
        "- Exercise-only control AUC **0.773** ≥ learned variants — see `reports/m3_results.md`.",
        "",
        "## RQ4 · Raw MM-Fit video",
        "",
        "- Four sessions (w00/w12/w13/w20); mean keeps sync≥70% only (`reports/rq4_first_look.md`).",
        f"- RQ4 run keys: `{list((rq4 or {}).keys())[:8]}` …" if rq4 else "- (no rq4 metrics yet)",
        "",
        "## Big data / coach",
        "",
        "- Spark sync scaling: see week1 / `experiments/*_spark_scaling` if present.",
        "- Kafka replay: pose 29.8 msg/s @ 48 ms; HR 0.79 msg/s @ 19 ms (`experiments/20261107_kafka_replay`).",
        f"- Kafka coach: median model **{coach_k.get('median_model_latency_ms')}** ms, "
        f"e2e **{coach_k.get('median_e2e_latency_ms')}** ms; "
        f"slow@rep1 still present={coach_k.get('slow_down_at_rep1_still_present')} "
        "(`experiments/20261107_coach_kafka`).",
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
