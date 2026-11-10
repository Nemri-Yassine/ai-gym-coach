# RQ4 · Models 1–2 on our own recordings

Gym station (iPhone portrait 720×1280 @ 30 fps + Galaxy Watch 7 HR), two people, one day.
Labels from `data/own_manifest.csv` (note wins over the exercise button). **Test data only — no training or threshold tuning.**

Eval sets: **20** (partner 8 + yassine 12). Test sets (excluded from main numbers): **2** (yassine 2026-11-09_s01 lunges; yassine 2026-11-10_s01 squats, note=`tt`).
Exercises covered in eval: 9 of 10 MM-Fit (no lunges except the test set). All form labels = correct.
RPE ignored (almost every set typed 10). Weight typed as body weight — ignored.

## Label corrections

Wrong button pressed on 3 eval sets; the note (and folder name for 2 unsaved sets) is the truth:

| Set | Button | Note / folder | Truth |
|---|---|---|---|
| `yassine/2026-11-10_s007/set01` | squats | PUSHU¨YASSINE | **pushups** |
| `yassine/2026-11-10_PUHSUP2/set01` | — | 2026-11-10_PUHSUP2 | **pushups** |
| `yassine/2026-11-10_JUMPJACK/set01` | — | 2026-11-10_JUMPJACK | **jumping_jacks** |
| `yassine/2026-11-10_TRICEPS/set01` | squats | TRICEPS | **tricep_extensions** |
| `yassine/2026-11-10_TRICEPS/set04` | lateral_shoulder_raises | BICEP 2 | **bicep_curls** |

## Model 1 (pose only)

Primary checkpoint: `m1_transformer_pose`. Windows 90 / hop 45, majority vote excluding `non_activity`.
When every window is `non_activity`, the vote is **undefined** → labelled `no_exercise_detected` and counted **wrong** (n=2 eval sets).
Set accuracy: **0.7** (14/20), Wilson 95% CI [0.48, 0.85]. Macro-F1 (9 exercises present): **0.511**.
Window accuracy: **0.629** (n=299).

Differences of a few sets are noise at this n.

| Person | n | Accuracy | Wilson 95% |
|---|---:|---:|---|
| yassine | 12 | 0.750 | [0.47, 0.91] |
| partner | 8 | 0.625 | [0.31, 0.86] |

| Exercise | n | Accuracy |
|---|---:|---:|
| bicep_curls | 2 | 1.000 |
| dumbbell_rows | 1 | 0.000 |
| dumbbell_shoulder_press | 4 | 1.000 |
| jumping_jacks | 2 | 1.000 |
| lateral_shoulder_raises | 1 | 0.000 |
| pushups | 4 | 1.000 |
| situps | 1 | 0.000 |
| squats | 2 | 1.000 |
| tricep_extensions | 3 | 0.000 |

### Wrong-button check

Did Model 1 agree with the note (truth) or the wrong button?

| Set | Button | Truth | M1 | Agrees with |
|---|---|---|---|---|
| `yassine/2026-11-10_s007/set01` | squats | pushups | pushups | **note/truth** |
| `yassine/2026-11-10_TRICEPS/set01` | squats | tricep_extensions | no_exercise_detected | **neither** |
| `yassine/2026-11-10_TRICEPS/set04` | lateral_shoulder_raises | bicep_curls | bicep_curls | **note/truth** |

### Comparison (same pose-only transformer, set/video accuracy)

| Source | Metric | Value | n |
|---|---|---:|---:|
| MM-Fit test_unseen (dataset pose) | macro-F1 | 0.85 | — |
| MM-Fit w20 MediaPipe | macro-F1 | 0.80 | — |
| RepCount public videos | video accuracy (4 ex.) | 0.808 | — |
| **Our videos** | set accuracy | 0.7 | 20 |
| **Our videos** | macro-F1 (9 ex.) | 0.511 | 20 |

Confusion: `reports/figures/m1_confusion_own.png`.

### Per-set predictions

| Set | use | Truth | transformer | CNN | LSTM | reps typed | reps verified | reps counted |
|---|---|---|---|---|---|---:|---:|---:|
| `partner/2026-11-10_s007/set01` | eval | pushups | pushups | pushups | pushups | 10 | — | 11 |
| `partner/2026-11-10_PUSHUP/set01` | eval | pushups | pushups | pushups | pushups | 10 | — | 14 |
| `partner/2026-11-10_EX3IHEB/set01` | eval | jumping_jacks | jumping_jacks | jumping_jacks | jumping_jacks | 10 | — | 11 |
| `partner/2026-11-10_SHOULDERPRESS/set01` | eval | dumbbell_shoulder_press | dumbbell_shoulder_press | dumbbell_shoulder_press | dumbbell_shoulder_press | 10 | — | 12 |
| `partner/2026-11-10_SHOULDERDUMB2/set01` | eval | dumbbell_shoulder_press | dumbbell_shoulder_press | dumbbell_shoulder_press | dumbbell_shoulder_press | 10 | — | 17 |
| `partner/2026-11-10_s01/set01` | eval | lateral_shoulder_raises | jumping_jacks | jumping_jacks | jumping_jacks | 10 | — | 11 |
| `partner/2026-11-10_s01/set02` | eval | tricep_extensions | dumbbell_rows | dumbbell_rows | dumbbell_rows | 10 | — | 15 |
| `partner/2026-11-10_s01/set03` | eval | situps | pushups | pushups | no_exercise_detected | 10 | — | 12 |
| `yassine/2026-11-10_s007/set01` | eval | pushups | pushups | pushups | pushups | 10 | — | 11 |
| `yassine/2026-11-10_PUHSUP2/set01` | eval | pushups | pushups | pushups | pushups | — | — | 11 |
| `yassine/2026-11-10_JUMPJACK/set01` | eval | jumping_jacks | jumping_jacks | jumping_jacks | jumping_jacks | — | — | 16 |
| `yassine/2026-11-10_DUMBSHOULD/set01` | eval | dumbbell_shoulder_press | dumbbell_shoulder_press | dumbbell_shoulder_press | dumbbell_shoulder_press | 10 | — | 6 |
| `yassine/2026-11-10_DUMBSHOULD/set02` | eval | dumbbell_shoulder_press | dumbbell_shoulder_press | dumbbell_shoulder_press | dumbbell_shoulder_press | 10 | — | 9 |
| `yassine/2026-11-10_SQUAT/set01` | eval | squats | squats | squats | squats | 10 | — | 10 |
| `yassine/2026-11-10_SQUAT/set02` | eval | squats | squats | squats | squats | 10 | — | 10 |
| `yassine/2026-11-10_TRICEPS/set01` | eval | tricep_extensions | no_exercise_detected | no_exercise_detected | no_exercise_detected | 10 | — | 7 |
| `yassine/2026-11-10_TRICEPS/set02` | eval | tricep_extensions | no_exercise_detected | bicep_curls | no_exercise_detected | 10 | — | 8 |
| `yassine/2026-11-10_TRICEPS/set03` | eval | bicep_curls | bicep_curls | bicep_curls | bicep_curls | 10 | — | 10 |
| `yassine/2026-11-10_TRICEPS/set04` | eval | bicep_curls | bicep_curls | bicep_curls | bicep_curls | 10 | — | 33 |
| `yassine/2026-11-10_DUMBLLROWS/set01` | eval | dumbbell_rows | pushups | pushups | pushups | 10 | — | 10 |
| `yassine/2026-11-09_s01/set01` | test | lunges | no_exercise_detected | no_exercise_detected | no_exercise_detected | 8 | — | 2 |
| `yassine/2026-11-10_s01/set01` | test | squats | squats | squats | situps | 10 | — | 14 |

## Rep counting

No peak-parameter tuning on these sets. **Main numbers use hand-verified reps only** (`reps_source=verified`).

> **PROVISIONAL:** `data/own_reps_verified.csv` is missing. Run `python scripts/review_reps.py` and count the 22 sets. Do not cite typed-default MAE as a result.


**Typed default (not a count):** MAE 3.17, exact 22.2%, ±1 50.0% (n=18). Station UI used to default reps=10 — these numbers are **not** ground truth.

Figure: `reports/figures/rep_counting_own.png`.

## Model 2 (form)

Only **correct** squats recorded (2 eval + 1 test). No lunges/plank mistakes. This measures **false alarms**, not mistake detection.
Eval squats: mean P(correct) **0.06**, % reps classified correct **0.0** (n=20).
Compare MM-Fit RQ4 MediaPipe P(correct) on squat/lunge: **0.08 / 0.14**.

## Heart rate

Eval sets only:

| Person | sets | hr_ok | mean coverage | median coverage | mean HR rise (end−start) |
|---|---:|---:|---:|---:|---:|
| yassine | 12 | 7 | 0.54 | 0.58 | 2.3 |
| partner | 8 | 4 | 0.52 | 0.59 | -1.5 |

Test sets (excluded above): n=2, hr_ok=0, mean coverage 0.25, median 0.25, mean HR rise –.

Limitation: the watch went silent for 20–80 s on several sets (hr_coverage 0 on many early sets).
Figure: `reports/figures/own_hr_session.png`.

## Limits

- Small n (20 eval sets), one day, two people, 9 exercises, no lunges in eval.
- Portrait phone framing and gym background differ from MM-Fit / RepCount.
- RPE not usable; HR gaps; two sets missing typed reps.
- Own data never used for training or threshold tuning in this report.
- Rep MAE typed-default is provisional until `data/own_reps_verified.csv` is filled.

Metrics: `C:/Users/yassi/OneDrive/Bureau/projet DL BIG DATA/experiments/20261110_own_rq4/metrics.json`.

## Why Model 1 fails on 4 exercises (diagnosis)

Figure: `reports/figures/own_m1_diagnosis.png` (mean key-joint Y vs rep fraction; our videos vs MM-Fit train true vs predicted).

- **Tricep extensions → `no_exercise_detected`:** MM-Fit triceps are overhead standing curls with wrists ~1.5–2 torso lengths above mid-hip; ours stay near 0–0.5 with little amplitude — likely a seated / kickback variant the pose model barely “sees” as exercise.
- **Lateral shoulder raises → jumping jacks:** Arm arcs look almost identical; MM-Fit jumping jacks add small ankle bounce. Ours keep ankles flat, but the model still keys on the arms.
- **Sit-ups → push-ups:** Our sit-ups keep ankles ~−1.2 below mid-hip (legs extended on the floor); MM-Fit sit-ups put ankles near the hip line. That floor framing plus noisy end-of-set motion overlaps the push-up wrist band.
- **Dumbbell rows → push-ups:** Ours are strongly one-sided (one wrist oscillating below the hip, the other flat) — body-horizontal row from the side. MM-Fit rows are more bilateral; the low active wrist looks like a push-up support arm.

Systematic, not random noise: camera view and exercise variant differ from MM-Fit.

## LOPO own adaptation (separate experiment)

Fine-tune pose-only transformer on MM-Fit train + one person's own eval sets; test on the other person. No threshold tuning on the test person. **Zero-shot remains the main RQ4 number.**

| Adapt on | Test on | n test sets | Zero-shot set acc | LOPO set acc | Checkpoint |
|---|---|---:|---:|---:|---|
| yassine | partner | 8 | 0.625 | **0.625** | `m1_transformer_pose_lopo_partner.pt` |
| partner | yassine | 12 | 0.75 | **0.917** | `m1_transformer_pose_lopo_yassine.pt` |

n = 8 (partner) / 12 (yassine) eval sets. Metrics: `C:/Users/yassi/OneDrive/Bureau/projet DL BIG DATA/experiments/20261110_m1_lopo_own_summary/metrics.json`.
