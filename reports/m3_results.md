# Model 3 results (MM-Fit test_unseen)

Train on MM-Fit train, early-stop on val, test on `test_unseen`. No fine-tune stage (no own recordings). No RPE: Spearman is predicted last-rep fatigue vs `set_idx_in_exercise` (later sets should look more tired).

**Limitation:** the breakdown label and the inputs share the speed signal (and B1 uses speed). Lead time only counts warnings that fire *before* breakdown.

| Variant | What | Risk AUC | Lead time (reps) | False alarms / session | Spearman(fatigue, set_idx) |
|---|---|---:|---:|---:|---:|
| B1 rule | stop when speed loss > 25% | 0.500 | 0.00 | 0.00 | — |
| B2 | general + HR | **0.884** | 0.93 | 0.00 | −0.14 |
| B3 | personal, no HR | 0.814 | 0.85 | 1.20 | −0.57 |
| **B4 (ours)** | personal + HR | 0.803 | 0.85 | 2.00 | −0.71 |
| B5 | general, no HR | 0.874 | 0.95 | 0.00 | −0.12 |

## Research questions (honest)

- **RQ2 (B4 vs B3):** HR does **not** help. B4 AUC 0.803 < B3 0.814. On w05 (no HR), B4 still runs (mask); AUC 0.845 with more false alarms (5.0).
- **RQ3 (B4 vs B2):** personal baseline does **not** beat general. B2 AUC 0.884 > B4 0.803. (Within-session baseline is weaker than a multi-week personal history would be.)
- **Existing apps (B4 vs B1):** B4 clearly beats the fixed 25% speed rule (0.803 vs 0.500 AUC; lead time 0.85 vs 0).

Negative Spearman: predicted end-of-set fatigue does **not** rise with later sets in the session under this weak label.

## Per-session (B4)

| Session | AUC | Lead | FA/session |
|---|---:|---:|---:|
| w00 | 0.760 | 1.33 | 0.0 |
| w05 (no HR) | 0.845 | 1.44 | 5.0 |
| w12 | 0.867 | 0.78 | 5.0 |
| w13 | 0.756 | 0.54 | 0.0 |
| w20 | 0.815 | 0.33 | 0.0 |
