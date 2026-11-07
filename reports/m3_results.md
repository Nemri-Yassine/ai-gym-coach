# Model 3 results · after Step 3.1b / 3.2b (MM-Fit test_unseen)

Train on MM-Fit train, early-stop on val, test on `test_unseen`. Labels are **speed-only** breakdown; risk stays 1 from breakdown−2 to end of set. No RPE. Three seeds (42, 43, 44); mean ± std. Alert threshold for lead/FA was chosen on **val** (Youden); the 0.7 column is for comparison. AUC does not depend on the threshold.

**Limitation:** breakdown and inputs still share the speed signal (and B1 uses speed). Lead time only counts warnings that fire *before* breakdown.

## Main table

| Variant | What | Risk AUC | Lead (Youden) | % warned ≥1 rep (Youden) | FA/sess (Youden) | Lead @0.7 | FA @0.7 | Spearman(fat, set_idx) | Spearman(fat, speed loss) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 rule | continuous speed-loss score; alert @25% | **0.751** | 0.37 | 17% | 0.2 | — | — | — | — |
| B1 @20% | same score; alert @20% (= label) | 0.751 | 0.51 | 24% | 4.2 | — | — | — | — |
| B2 | general + HR | 0.729 ± 0.029 | 1.02 | 37% | 45.8 | 0.02 | 1.1 | −0.08 | −0.16 |
| B3 | personal, no HR | 0.689 ± 0.026 | 0.59 | 18% | 35.3 | 0.15 | 4.7 | −0.61 | −0.08 |
| **B4 (ours)** | personal + HR | 0.641 ± 0.022 | 1.09 | 34% | 52.6 | 0.15 | 6.4 | −0.67 | −0.09 |
| B5 | general, no HR | 0.760 ± 0.009 | 0.55 | 21% | 33.1 | 0.02 | 0.3 | −0.07 | −0.11 |
| Exercise-only | one-hot exercise + rep_idx/n | **0.773** | — | — | — | — | — | — | — |

B1 alert rate @25%: **27%** of test sets (was falsely reported as AUC 0.50 / lead 0 / FA 0 under the old risk window).

Youden thresholds (~0.5) give high FA; the **0.7** column is the usable operating point for a coach (low lead, lower FA). AUC is the fair ranking metric.

## Research questions (honest)

- **RQ2 (B4 vs B3):** HR does **not** help. B4 0.641 < B3 0.689.
- **RQ3 (B4 vs B2):** personal within-session baseline does **not** beat general. B2 0.729 > B4 0.641.
- **Existing apps (B4 vs B1):** with a fair continuous B1, **B4 does not beat B1** (0.641 vs 0.751). The old “B4 beats apps” claim is **withdrawn**.
- **Exercise-only control:** logistic regression on exercise + position reaches **0.773**, above every learned variant. Model 3 is **not** clearly learning fatigue beyond exercise identity and rep position on these labels.

## Spearman sign

`Spearman(last-rep fatigue, set_idx)` is negative (B4 −0.67). Diagnostics on test_unseen:

- mean reps by set_idx ≈ 9.45 / 9.37 / 9.48 — later sets are **not** shorter
- Spearman(n_reps, set_idx) ≈ −0.06 — almost none

So the negative sign is **not** explained by shorter later sets. Spearman vs measured end-of-set speed loss is also near zero / negative (−0.09 for B4): predicted fatigue does not track real speed loss. Likely the fatigue head mostly reflects within-set position, and personal normalisation further distorts set-to-set comparison.

## w05 (B4, no HR)

Mask still works: AUC 0.84 on w05 alone under the Youden threshold (high FA). Graceful degradation of the HR channels, not a clean win.
