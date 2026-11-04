# Model 2 benchmark

Leave-one-person-out on EC3D, four people. Every model is scored on the same test reps, projected at 45° with no mirror and no augmentation. The random forest uses the six 2D features of that same skeleton.

The ± value is the sample standard deviation across the four folds (ddof=1). With four people, one person moves the mean a lot.

Authors' graph network: 90.9%. Different input and setup, so this is context only, not a target for the table.

Earlier 3D random forest on squats: 53%. That number is from the earlier analysis, not from this benchmark.

## Accuracy at 45° (mean ± std, %)

| Model | Squat | Lunge | Plank | Train minutes |
|---|---:|---:|---:|---:|
| stgcn | 55.6 ± 12.4 | 68.5 ± 8.2 | 70.2 ± 19.1 | 73.34 |
| lstm | 33.3 ± 11.5 | 61.3 ± 15.7 | 64.0 ± 18.7 | 14.92 |
| rf | 53.8 ± 6.1 | 71.6 ± 19.3 | 82.8 ± 7.3 | 0.19 |

## Macro-F1 at 45° (mean ± std, %)

| Model | Squat | Lunge | Plank |
|---|---:|---:|---:|
| stgcn | 53.1 ± 16.0 | 63.5 ± 10.6 | 66.2 ± 20.8 |
| lstm | 28.6 ± 9.3 | 55.6 ± 20.4 | 58.4 ± 21.4 |
| rf | 46.3 ± 10.1 | 69.0 ± 20.4 | 81.5 ± 11.4 |

## Mistake detection at 45°

Positive class = not correct. Specificity is the share of correct reps recognized as correct. A model that always says "mistake" has recall 100% and specificity 0%. Majority accuracy predicts the most common training class of that exercise.

| Model | Exercise | Recall | Precision | Specificity | Majority accuracy | Always-mistake precision |
|---|---|---:|---:|---:|---:|---:|
| stgcn | squat | 81.5 ± 30.9 | 79.0 ± 15.9 | 40.0 ± 49.0 | 31.3 ± 3.5 | 68.7 ± 3.5 |
| stgcn | lunge | 77.9 ± 27.0 | 83.9 ± 18.6 | 60.4 ± 45.8 | 36.2 ± 1.6 | 63.8 ± 1.6 |
| stgcn | plank | 82.1 ± 20.6 | 83.2 ± 16.4 | 52.7 ± 47.7 | 39.6 ± 5.8 | 68.0 ± 2.9 |
| lstm | squat | 72.6 ± 23.4 | 73.0 ± 18.0 | 30.0 ± 46.9 | 31.3 ± 3.5 | 68.7 ± 3.5 |
| lstm | lunge | 55.5 ± 30.9 | 84.7 ± 11.5 | 76.1 ± 19.8 | 36.2 ± 1.6 | 63.8 ± 1.6 |
| lstm | plank | 80.9 ± 15.8 | 73.3 ± 13.6 | 34.5 ± 36.3 | 39.6 ± 5.8 | 68.0 ± 2.9 |
| rf | squat | 82.1 ± 19.6 | 85.8 ± 15.1 | 62.5 ± 45.0 | 31.3 ± 3.5 | 68.7 ± 3.5 |
| rf | lunge | 73.8 ± 18.9 | 85.5 ± 17.9 | 77.1 ± 28.4 | 36.2 ± 1.6 | 63.8 ± 1.6 |
| rf | plank | 86.7 ± 16.3 | 91.7 ± 11.8 | 74.0 ± 40.7 | 39.6 ± 5.8 | 68.0 ± 2.9 |

Correct share in this run's rep table: squat 31.1%, lunge 36.2%, plank 32.0%.

## Per fold, accuracy at 45° (%)

| Model | Exercise | Hugues | Isinsu | Sena | Vidit |
|---|---|---:|---:|---:|---:|
| stgcn | squat | 74.2 | 50.0 | 50.0 | 48.4 |
| stgcn | lunge | 59.4 | 64.5 | 78.1 | 71.9 |
| stgcn | plank | 68.2 | 91.3 | 45.5 | 76.0 |
| lstm | squat | 32.3 | 47.4 | 34.4 | 19.4 |
| lstm | lunge | 65.6 | 45.2 | 53.1 | 81.2 |
| lstm | plank | 86.4 | 52.2 | 45.5 | 72.0 |
| rf | squat | 51.6 | 55.3 | 46.9 | 61.3 |
| rf | lunge | 50.0 | 67.7 | 96.9 | 71.9 |
| rf | plank | 77.3 | 87.0 | 90.9 | 76.0 |

## Accuracy by camera angle (mean ± std, %)

Same test reps, no mirror. 45° is the main result. The other angles are the robustness check.

| Model | Exercise | 0° | 15° | 30° | 45° | 60° | 75° | 90° |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stgcn | squat | 58.9 ± 10.6 | 54.0 ± 9.2 | 57.0 ± 11.9 | 55.6 ± 12.4 | 49.5 ± 11.0 | 47.2 ± 16.2 | 48.1 ± 22.5 |
| stgcn | lunge | 73.2 ± 6.5 | 74.0 ± 9.0 | 72.4 ± 7.9 | 68.5 ± 8.2 | 68.5 ± 8.2 | 66.9 ± 5.8 | 64.5 ± 8.3 |
| stgcn | plank | 69.5 ± 20.4 | 70.2 ± 19.1 | 69.5 ± 20.4 | 70.2 ± 19.1 | 69.5 ± 19.1 | 69.5 ± 20.4 | 69.5 ± 20.4 |
| stgcn | overall | 66.2 ± 5.7 | 65.0 ± 3.7 | 65.4 ± 5.3 | 63.7 ± 4.1 | 61.2 ± 3.3 | 59.8 ± 5.4 | 59.4 ± 8.6 |
| lstm | squat | 31.3 ± 7.4 | 35.0 ± 8.9 | 31.4 ± 10.8 | 33.3 ± 11.5 | 37.8 ± 3.8 | 40.1 ± 4.6 | 38.6 ± 3.1 |
| lstm | lunge | 51.2 ± 16.0 | 53.5 ± 15.8 | 57.4 ± 17.4 | 61.3 ± 15.7 | 58.9 ± 19.3 | 57.4 ± 17.3 | 55.8 ± 11.7 |
| lstm | plank | 66.0 ± 6.8 | 68.8 ± 16.1 | 63.4 ± 19.4 | 64.0 ± 18.7 | 76.1 ± 16.5 | 77.8 ± 14.9 | 81.8 ± 10.7 |
| lstm | overall | 48.1 ± 6.7 | 51.0 ± 5.2 | 49.8 ± 7.3 | 51.9 ± 7.0 | 55.9 ± 8.0 | 56.7 ± 9.6 | 56.9 ± 5.3 |
| rf | squat | 48.9 ± 10.4 | 52.1 ± 10.5 | 56.8 ± 12.7 | 53.8 ± 6.1 | 57.7 ± 6.7 | 51.2 ± 16.4 | 42.2 ± 15.4 |
| rf | lunge | 51.2 ± 14.1 | 52.8 ± 10.7 | 61.4 ± 20.4 | 71.6 ± 19.3 | 81.8 ± 12.4 | 79.5 ± 12.2 | 83.4 ± 12.0 |
| rf | plank | 46.1 ± 7.3 | 48.3 ± 10.6 | 72.2 ± 4.5 | 82.8 ± 7.3 | 84.3 ± 10.0 | 82.3 ± 5.4 | 82.2 ± 4.6 |
| rf | overall | 48.8 ± 6.3 | 51.3 ± 5.6 | 62.8 ± 7.5 | 68.2 ± 8.5 | 73.7 ± 4.9 | 69.8 ± 9.4 | 67.8 ± 6.2 |

## Random forest on 3D features (separate)

3D features, one vector per rep, same people held out. Not the 2D comparison.

| Exercise | Accuracy (mean ± std, %) |
|---|---:|
| squat | 55.8 ± 24.2 |
| lunge | 82.6 ± 11.1 |
| plank | 91.5 ± 5.1 |

## Checkpoints

Each run also saves a checkpoint trained on all four people (`models/`). That checkpoint is for live use. It is not scored, because it has seen every person.

| Model | LOSO train minutes | All-people minutes | Parameters | Checkpoint | Source |
|---|---:|---:|---:|---|---|
| stgcn | 73.34 | 24.29 | 245451 | `m2_stgcn.pt` | `C:\Users\yassi\OneDrive\Bureau\projet DL BIG DATA\experiments\20261104_m2_stgcn\metrics.json` |
| lstm | 14.92 | 5.05 | 53195 | `m2_lstm.pt` | `C:\Users\yassi\OneDrive\Bureau\projet DL BIG DATA\experiments\20261104_m2_lstm\metrics.json` |
| rf | 0.19 | 0.04 | — | `m2_rf.pickle` | `C:\Users\yassi\OneDrive\Bureau\projet DL BIG DATA\experiments\20261104_m2_rf\metrics.json` |
