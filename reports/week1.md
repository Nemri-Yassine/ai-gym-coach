# Week 1 validation

What was built: a Spark job that joins MM-Fit pose, watch IMU and heart rate into one 30 Hz Parquet table; one 18-joint skeleton shared by MM-Fit, EC3D and (later) MediaPipe; 3-second windows; a 9-run exercise-recognition benchmark; and a rep counter. The phone/watch recorder and the recording protocol are ready. No own session has been filmed yet.

## Spark

21 sessions, **1,460,366** rows. 99.8% of frames have watch IMU. 81.1% have heart rate, because w05, w10 and w18 have no heart-rate file.

Each session is its own Spark task (adaptive coalescing was collapsing all 21 into one task, which hid any effect of extra cores).

| Cores | Runtime |
|---:|---:|
| 1 | 57.7 s |
| 2 | 40.9 s |
| 4 | 37.1 s |

Going from 1 to 2 cores helps. A fourth core barely does: after two cores the job is mostly reading `.npy` files and writing Parquet. Figure: `reports/figures/spark_runtime.png`.

## Model 1 (RQ1, first answer)

Best run: **transformer, pose + IMU**. On `test_unseen` (workouts w00, w05, w12, w13, w20): macro-F1 **0.880**, accuracy **0.950**. The target was 0.85. The same run on people who were seen in training is macro-F1 0.975. 166k parameters, 24.7 minutes on CPU.

Pose alone is weaker (transformer 0.836, CNN 0.785). IMU alone is strong for the CNN (0.842) and weak for the LSTM and the transformer (about 0.75). Using both is the best cell in every architecture. Full table: `reports/m1_benchmark.md`. Confusion matrix: `reports/figures/m1_confusion_test_unseen.png`.

The classes the best model still misses on unseen people: dumbbell rows (F1 0.59), squats (0.71), situps (0.79). The other eight classes are above 0.90.

32,421 windows. Train is dominated by `non_activity` (11,628 of 15,756). Class weights are inverse frequency.

## Rep counting

PCA of the 36 joint coordinates, smoothed, peaks at least 0.6 s apart. **616** sets. Macro MAE **1.35** reps (target 0.50). Micro MAE 1.34.

| Strong | MAE | Weak | MAE |
|---|---:|---|---:|
| squats | 0.28 | bicep curls | 5.07 |
| jumping jacks | 0.44 | lunges | 3.10 |
| dumbbell rows | 0.53 | pushups | 1.37 |

Squats, rows, presses, jacks, raises and tricep extensions are within ±1 rep on at least 98% of sets. Bicep curls are never within ±1: the main component is not one clean peak per curl. Lunges are the other miss (exact on 29% of sets). An earlier hip-height counter was exact on squats and lunges; this one is exercise-agnostic, which is what the plan asked for, and it pays for that on arm work and lunges. Figure: `reports/figures/rep_counting_mae.png`.

## Our sessions

**0 recorded.** The recorder (`scripts/record_session.py`) and `docs/recording_protocol.md` are in place. Pulsoid, HypeRate and Sensor Logger are implemented. None was tried on the Poco X3 Pro or the Galaxy Watch 7, so there is no latency number and the heart-rate choice is still open (`docs/decisions.md`). Week 3 still needs at least 6 normal sessions and 2 form sessions per person. Recording has to start before Model 3.

## Problems

- PySpark 4.2 warns that pandas 3.1 is not fully supported. The Parquet round-trip and the 21-session job both succeeded anyway.
- The project path contains spaces. Spark on Windows splits unquoted paths, so the virtualenv is exposed through a junction at `C:\awc_tmp\venv`, and the Parquet output through `C:\awc_tmp\links`.
- Rep counting misses its MAE target. Left as measured. Not retuned.
