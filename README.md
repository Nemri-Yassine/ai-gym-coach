# AI Workout Coach

**Most apps tell you what you did. Ours tells you when to stop.**

An AI coach that watches you train with a **phone camera** and a **smartwatch** (Galaxy Watch 7). It recognizes the exercise, counts reps, checks the form of every rep, and compares today's session with **your own past sessions** to say *stop* before fatigue turns into injury.

Course project · Deep Learning & Big Data · 2026–2027

**Team:** Yassine Nemri ([@Nemri-Yassine](https://github.com/Nemri-Yassine)) · Iheb Lamouchi ([@iheblam](https://github.com/iheblam))

---

## Progress

The step-by-step work plan (3 weeks, one validation per week) is in [docs/PLAN.md](docs/PLAN.md), with a designed PDF in `reports/Implementation_Plan.pdf`. Cursor follows it with the rules in `.cursor/rules/awc.mdc`.

| Step | What | Status |
|---|---|---|
| 0 | Project proposal and validation presentation | ✅ Done · Oct 7, 2026 |
| 1 | Project structure, git, dataset loaders | ✅ Done · Oct 7, 2026 |
| 1.1 | Environment, Spark helper, run tracking | ✅ Done · Oct 22, 2026 |
| 1.2 | Capture decision, recorder, recording protocol | ✅ Done · Nov 10, 2026 · own data + station |
| — | Gym station (iPhone Wi-Fi → laptop + watch HR) | ✅ Done · Nov 9, 2026 |
| 1.3 | Spark sync of MM-Fit to Parquet | ✅ Done · Oct 22, 2026 |
| 1.4 | One skeleton for every source, Model 1 windows | ✅ Done · Oct 22, 2026 |
| 1.5 | Model 1 benchmark and rep counting | ✅ Done · Oct 22, 2026 |
| 1.6 | Week 1 validation package | ✅ Done · Oct 22, 2026 |
| 2.1 | EC3D rep dataset for Model 2 | ✅ Done · Nov 4, 2026 |
| 2.2 | Model 2 benchmark on EC3D | ✅ Done · Nov 4, 2026 |
| 2.3 | Kafka streaming of pose and HR (replay) | ✅ Done · Nov 7, 2026 |
| 2.4 | Models 1–2 on raw MM-Fit video + own (RQ4) | ✅ Done · Nov 7 / Nov 10, 2026 |
| 2.4a | Own-recordings manifest (note-corrected labels) | ✅ Done · Nov 10, 2026 |
| 2.4b-own | MediaPipe pose for own recordings | ✅ Done · Nov 10, 2026 |
| 2.4c | Models 1–2 on own recordings (RQ4 own) | ✅ Done · Nov 10, 2026 |
| 2.4b | RQ4 on four MM-Fit videos + auto sync | ✅ Done · Nov 7, 2026 |
| 2.5 | Spark rep history + within-session baselines | ✅ Done · Nov 7, 2026 |
| 2.5b | Spark rep table for own recordings | ✅ Done · Nov 10, 2026 |
| 2.6 | Week 2 validation package | ✅ Done · Nov 7, 2026 · refreshed Nov 10 |
| 2.6b | Station post-set detection (live demo) | ✅ Done · Nov 10, 2026 |
| 3.1 | Model 3 sequences and labels (MM-Fit) | ✅ Done · Nov 7, 2026 |
| 3.1b | Speed-only breakdown; risk held to end of set | ✅ Done · Nov 7, 2026 |
| 3.2 | Model 3 + RQ2/RQ3 comparisons | ✅ Done · Nov 7, 2026 · superseded by 3.2b |
| 3.2b | Fair B1, val thresholds, exercise-only control | ✅ Done · Nov 7, 2026 |
| 3.3 | Replay coach (text; voice optional) | ✅ Done · Nov 7, 2026 |
| 3.3b | Replay coach through Kafka (e2e latency) | ✅ Done · Nov 7, 2026 |
| 3.4 | Evaluation and final report | ✅ Done · Nov 7, 2026 · refreshed for v1.1 |
| 3.5 | Rehearsal + v1.0 tag | ✅ Done · Nov 7, 2026 |
| — | Final report PDF + week 2–3 slides + v1.1 | ✅ Done · Nov 7, 2026 |

### Log

- **2026-11-10 · Fix 5 (M1 diagnosis + LOPO).** Trajectory figure `own_m1_diagnosis.png`; RQ4 notes on triceps/lateral/situps/rows vs MM-Fit. LOPO adaptation (MM-Fit train + one person → other): partner **0.625 → 0.625** (n=8), yassine **0.75 → 0.917** (n=12). Zero-shot remains the main RQ4 result. Checkpoints `models/m1_transformer_pose_lopo_*.pt`.
- **2026-11-10 · Fix 4 (report hygiene).** Rounded RQ4 numbers (macro-F1 **0.511**); HR/Spark tables eval-only; `no_exercise_detected` when all windows are non_activity (n=2 eval); per-set table in `rq4_own.md`; clock RTT medians **3.5 ms** (not “tens of ms”); Spark `segmentation_failed` (4 sets) excluded from velocity medians.
- **2026-11-10 · Fix 3 (honest live latency).** Cached “0.84 s” skipped MediaPipe. Full pose mean **13.1 s**; live_fast (every 2nd frame + 480px) mean **7.8 s**, exercise match 3/3. Form hidden until calibrated (`--show-form`). Docker Kafka daemon down — not claimed. `experiments/20261110_station_latency/`.
- **2026-11-10 · Fix 2 (test isolation).** Quick-trains honour `AWC_MODELS_DIR` / `AWC_EXPERIMENTS_DIR` and write `*_quick` checkpoints under tmp in pytest. Stray quick folders removed; B5 retrain: risk AUC **0.599 → 0.760 ± 0.009** (matches `reports/m3_results.md`).
- **2026-11-10 · Fix 1 (verified reps).** Station no longer defaults reps/RPE to 10; Save set requires a typed count. Review UI: `python scripts/review_reps.py` → `data/own_reps_verified.csv`. Manifest keeps `reps_typed` / `reps_source`; eval MAE uses **verified only**. Typed-default MAE **3.17** is provisional (not a count) until review finishes.
- **2026-11-10 · Weeks 1–2 complete (own data).** Gym station capture decision locked; week1/week2 reports refreshed; slides updated; tag `v1.2`. Key RQ4 own: M1 set acc **0.70** [0.48, 0.85], macro-F1 **0.51**, rep MAE **3.17** (typed default — provisional), M2 P(correct) **0.06**, Spark own **269** reps, live analysis **0.84/0.03/0.03 s**.
- **2026-11-10 · Step 2.6b.** Station post-set analysis (background): pose → M1 → reps → M2 → HR → `data/interim/own_live/…/analysis_setNN.json`; shown on the control page (never into `sets.csv`). Latency on 3 cached sets: **0.84 / 0.03 / 0.03 s** (cold model load then warm). Optional `--kafka`. Demo: `docs/demo.md`.
- **2026-11-10 · Step 2.5b.** Own Spark rep table: **269** rows, 22 sets, 2 people → `data/interim/own_rep_table.parquet` (speed vs first 2 reps, velocity loss). `python -m awc.bigdata.history --own`.
- **2026-11-10 · Step 2.4c.** RQ4 on own videos (pose-only transformer): set accuracy **0.70** (14/20, Wilson 95% [0.48, 0.85]), macro-F1 **0.51**; yassine 0.75 / partner 0.625; rep MAE **3.17** (exact 22%, ±1 50%, n=18); M2 mean P(correct) on correct squats **0.06** (false-alarm check). Wrong-button: M1 agreed with note on 2/3. Report: `reports/rq4_own.md`.
- **2026-11-10 · Step 2.4b-own.** MediaPipe pose for **22**/22 own sets → `data/interim/own/`; mean person-detect **95.7%** (Nov 9 lunges test set 22% — poor framing); HR join ±2 s; figure `reports/figures/own_pose_check.png`. `python scripts/process_recording.py --all --workers 4`.
- **2026-11-10 · Step 2.4a.** Own-recordings manifest `data/own_manifest.csv`: **22** sets (20 eval + 2 test), 2 people, 9 exercises; note wins over button (3 wrong-button corrections); HR coverage + `hr_ok` per set. RPE/weight ignored (RPE≈10, body weight typed). Rebuild: `python scripts/own_manifest.py --print`.
- **2026-11-09 · Gym station.** iPhone Safari → laptop over HTTPS/WebSocket (`python -m awc.station`); live chunk upload to `data/raw/own_recordings/`; BLE or Pulsoid HR; clock sync; control UI + QR. Docs: `docs/gym_station.md`. Check: `scripts/station_check.py`.
- **2026-11-10 · Gym station v2.** Certificate re-issued automatically at start for the current IPs + iPhone hotspot range (same root CA, nothing new on the phone); 4-digit PIN for the phone page, control page laptop-only. Video uploaded live again (ordered retry queue, `set_done` after the last piece), remuxed to a normal MP4, frame times from the video's own timestamps + recorder start (yesterday's real iPhone clip: 332 frames, 30.0 fps; the old 360-vs-332 mismatch is gone). Smooth live view via WebRTC with JPEG fallback. Faster BLE reconnect + drop counter (Nov 9 link dropped about every 30 s). Firewall rule must allow Public networks.
- **2026-11-07 · Final report + week 2–3 slides.** `reports/AI_Workout_Coach_Final_Report.pdf`, `reports/validation2/AI_Workout_Coach_Week2-3.html`. Tag `v1.1`.
- **2026-11-07 · Step 2.4b.** RQ4 on w00/w12/w13/w20 with auto sync. Only **w20** sync≥70% (10/12); w00/w12/w13 marked † and excluded from mean. Auto vs measured on w20: mean |Δ| **1.42** frames. Table: `reports/rq4_first_look.md`.
- **2026-11-07 · Step 3.3b.** Coach consumes Kafka `pose.keypoints` / `watch.hr`, publishes `coach.events`. Two w20 demos (seek frame 2100): median **model 0.83 ms**, median **e2e 75 ms** (p90 e2e ~831 ms). **Slow down at rep 1 still present** (early risk ~0.40–0.45 > slow_thresh). Commands: `docs/demo.md`. Metrics: `experiments/20261107_coach_kafka/`.
- **2026-11-07 · Step 3.2b.** Fair B1 continuous AUC **0.751** (was 0.500). B4 **0.641 ± 0.022** — does **not** beat B1 or exercise-only control (**0.773**). RQ2/RQ3 still negative for B4. Old “beats apps” claim withdrawn. Table: `reports/m3_results.md`.
- **2026-11-07 · Step 3.1b.** Breakdown is **speed-only** (`use_form=False`); risk stays 1 from breakdown−2 to end of set. Squats breakdown rate **0.42** (was ~100% with form). Extreme: **jumping_jacks** rate **0.018** (< 5%) — reported here, not hidden. Stats: `data/processed/m3/breakdown_stats.json`.
- **2026-11-07 · Step 3.5.** Fresh `pytest -q` green; coach replay run twice; tag `v1.0`.
- **2026-11-07 · Step 3.4.** `reports/results.md` + `reports/final_report.md` (limits: no own data, no RPE, within-session baseline, shared speed label, watch untested, MM-Fit reps assumed correct).
- **2026-11-07 · Step 3.3.** Replay coach on w20 (text, `--no-voice`): median latency rep→message **< 1 ms** offline; voice path present but optional. `experiments/20261107_coach_replay/`.
- **2026-11-07 · Step 3.2.** Model 3 B1–B5 on test_unseen: B4 AUC **0.803** (does **not** win). RQ2: B4 0.803 vs B3 0.814; RQ3: B4 0.803 vs B2 0.884; B4 beats B1 (0.50). Table: `reports/m3_results.md`.
- **2026-11-07 · Step 3.1.** Model 3 set sequences from MM-Fit only: 615 sets (train 300 / val 86 / test_unseen 139). Labels: fatigue position, breakdown (speed −20% or form_score < 0.5), risk (breakdown within 2 reps). General + personal norms; no RPE; causal features.
- **2026-11-07 · Step 2.6.** Week 2 package in `reports/week2.md` (M2 conclusion, Kafka, RQ4 w20, rep table, data-source change).
- **2026-11-07 · Step 2.5.** Spark rep table: **5,918** MM-Fit reps, 616 sets; within-session baselines (`Window.partitionBy(session, exercise).orderBy(set_idx).rowsBetween(unboundedPreceding, -1)`); `has_baseline` rate 0.67. No RPE. Cross-session pseudo-users skipped (time). Summary: `data/processed/rep_table_summary.json`.
- **2026-11-07 · Step 2.4.** RQ4 on raw MM-Fit `w20` RGB → MediaPipe vs dataset pose (extra `test_unseen` videos w00/w12/w13 left downloading; shrunk to w20). Sync: 8/10 sets within 3 frames. M1 macro-F1 pose-only **0.85 → 0.80**; pose+IMU **0.66 → 0.48**. Rep MAE **0.93 → 1.56**. Model 2 RF mean P(correct) on squat/lunge (no form labels): 0.08 / 0.14. Table: `reports/rq4_first_look.md`.
- **2026-11-07 · Step 2.3.** Kafka via Docker `apache/kafka` (KRaft binaries blocked: `wmic` missing on this Windows). Replay of `w20_rgb.mp4` + `w20_sw_l_hr`: pose **29.8 msg/s** median latency **48 ms**; HR **0.79 msg/s**, **19 ms**. Sink writes under `data/interim/replay/` only. Live Galaxy Watch link is optional and **untested**. Setup: `docs/kafka_setup.md`. Metrics: `experiments/20261107_kafka_replay/`.
- **2026-11-07 · Plan revision.** One teammate was injured, so we have **no own sessions** and will not invent any. Validation uses MM-Fit RGB video (Zenodo 7672767) plus watch IMU/HR instead; personal baselines become within-session (earlier sets of the same exercise); RPE is dropped everywhere. Details in `docs/PLAN.md` §6.
- **2026-11-04 · Step 2.2.** Model 2 LOSO at 45° (mean ± std): ST-GCN squat **55.6 ± 12.4%**, lunge 68.5 ± 8.2, plank 70.2 ± 19.1; 2D random forest squat **53.8 ± 6.1%**, lunge 71.6 ± 19.3, plank 82.8 ± 7.3; angle LSTM squat 33.3 ± 11.5. ST-GCN beats the 2D RF on squat and clears the old 53% mark, but the RF wins on lunge and plank. Same-setup 3D RF (separate row): squat 55.8, lunge 82.6, plank 91.5. Authors' 90.9% is context only. Train minutes: ST-GCN 73.3, LSTM 14.9, RF 0.2. Table in `reports/m2_benchmark.md`.
- **2026-11-04 · Step 2.1.** EC3D rep dataset: 362 reps after dropping 9 unknown squats (squat 132, lunge 127, plank 103). 4 people, leave-one-person-out. Reps are 24–208 frames long (median 78), stored once in `data/processed/m2_reps.npz` (7,732,955 bytes) and resampled to 64 frames in the dataset.
- **2026-10-24 · Gym recorder (step 1.2, phone side).** Phone web app `docs/gym/index.html` (Chrome on the Poco X3 Pro): films each set, logs Galaxy Watch 7 heart rate over Bluetooth for the whole session (watch app *Heart for Bluetooth*; Pulsoid as fallback), RPE / reps / form tapped after each set, everything on the phone clock, saved to Downloads. `scripts/import_phone_session.py` converts a session into `data/raw/own_recordings/`. Tested end to end in demo mode: real Chrome MP4 at 30.2 fps, frame clock within one frame. Not yet tried on the real watch. Guide: `docs/gym_guide.md`. 42 tests pass.
- **2026-10-24 · Fixes after review.** Rep counter: lunges and curls were counted as half (alternating limbs); the counter now uses an exercise-specific signal. Macro MAE **1.35 → 0.49** on 616 MM-Fit sets (target 0.50 met); lunges 3.10 → 0.02, curls 5.07 → 0.46; public videos 3.67 → 1.95. Recorder: reader thread timestamps frames on arrival, video written at the measured fps, filming no longer pauses for set details (RPE is a key press, the rest is typed after the session); `extract_pose(..., frame_times_ms=...)` uses `frames.csv` as the clock. 40 tests pass. Exact package versions in `requirements-lock.txt`.
- **2026-10-23 · Public videos (stand-in for step 1.2 sessions).** 135 RepCount-A gym videos through MediaPipe → Model 1 → rep counter (`reports/public_videos.md`). Pose-only transformer: 80.8% of videos right on squats, pushups, situps and jumping jacks. The pose + IMU transformer falls to 67.9% without the watch. Rep counting MAE: squats 1.22, pushups 8.19. Own recordings are still needed before week 3.
- **2026-10-22 · Step 1.6.** Validation package in `reports/week1.md`. Best Model 1 is a transformer on pose + watch IMU: macro-F1 **0.880** on people it has never seen (target 0.85), accuracy 0.950. Rep counting macro MAE **1.35** (target 0.50): squats 0.28, bicep curls 5.07. Spark sync of 1,460,366 rows: 57.7 s / 40.9 s / 37.1 s on 1 / 2 / 4 cores.
- **2026-10-22 · Step 1.5.** Nine Model 1 runs in `reports/m1_benchmark.md`. Unseen macro-F1 from 0.746 (LSTM, IMU only) to 0.880 (transformer, both). Rep counting on all 616 MM-Fit sets in `reports/rep_counting.md`.
- **2026-10-22 · Step 1.4.** 32,421 windows of 90 frames (train 15,756, val 3,468, test seen 5,411, test unseen 7,786). All 11 classes are in the train split. `non_activity` is 11,628 of the train windows.
- **2026-10-22 · Step 1.3.** Synced Parquet: 21 sessions, 1,460,366 rows, 99.8% of frames have watch IMU, 81.1% have heart rate (w05, w10 and w18 have none).
- **2026-10-22 · Step 1.2.** Recorder and protocol are in place. Heart-rate source is not chosen yet: the phone and the watch were not connected, so no latency was measured. Own sessions recorded: **0**.
- **2026-10-22 · Step 1.1.** Temurin 17.0.20.1, Hadoop winutils 3.3.6, Spark helper, experiment folders. `pytest`: 26 passed.
- **2026-10-22 · Planning.** Three-week implementation plan with tests and acceptance checks per step (`docs/PLAN.md`, rebuild the PDF with `python scripts/docs/build_plan_pdf.py`).
- **2026-10-07 · Step 1.** Clean project structure, `awc` Python package, MM-Fit and EC3D loaders, data sanity tests, analysis scripts moved to `scripts/analysis/`; one-page overview in `reports/Project_Structure.pdf`.
- **2026-10-07 · Step 0.** Validation presentation (`presentation/index.html`) and briefing (`reports/AI_Workout_Coach_Briefing.pdf`). First measurements on the real data, see *Results so far*.

---

## How it works

```
Phone camera ──► skeleton (18 joints, MediaPipe)  ─┐
Galaxy Watch 7 ─► heart rate + wrist motion  ───────┤  Kafka (live)
                                                    ▼
          Model 1: which exercise, where each rep starts/ends
                                                    ▼
          Model 2: is this rep good, or which mistake?
                                                    ▼
          Model 3: fatigue / injury risk vs your own history   ◄── Spark (session history)
                                                    ▼
               Voice: "Keep going" · "Slow down" · "Stop"
```

## Research questions

1. **RQ1**: Does a model trained on public data recognize exercises and count reps on people it has never seen?
2. **RQ2**: Does the watch's heart rate detect fatigue better than movement alone?
3. **RQ3**: Does a personal baseline (your own past sessions) warn earlier than a general model? *(main contribution)*
4. **RQ4**: Do the models still work on our own phone and watch?

## Data

| Dataset | Job | Size |
|---|---|---|
| [MM-Fit](https://mmfit.github.io) | Exercise, reps, heart rate | 21 sessions, 6,160 reps, 70.8 M sensor rows |
| [EC3D](https://github.com/Jacoo-Zhao/3D-Pose-Based-Feedback-For-Physical-Exercises) | Good vs bad reps | 371 reps, 29,789 3D frames, 8 mistakes |
| Our recordings | RQ4 / fatigue (later) | 2 people, 20 eval sets recorded Nov 10 (more weeks still to collect) |

Download instructions: [data/README.md](data/README.md).

## Project structure

```
.
├── README.md              this file, updated at every step
├── requirements.txt       Python dependencies
├── pyproject.toml         makes src/awc installable, configures pytest
├── data/
│   ├── raw/               original datasets, never modified (not in git)
│   │   ├── mm-fit/        w00 … w20
│   │   ├── ec3d/          data_3D.pickle, data.pickle
│   │   └── own_recordings/
│   ├── interim/           synced + normalized streams (Parquet)
│   └── processed/         model-ready windows and rep tensors
├── src/awc/               the project's Python package
│   ├── config.py          all paths and constants
│   ├── data/              loaders: mmfit.py, ec3d.py, recordings.py
│   ├── features/          sync, skeleton normalization, windows, rep segmentation
│   ├── models/            m1_recognition.py, m2_form.py, m3_fatigue.py
│   ├── training/          training loops
│   ├── evaluation/        metrics and benchmarks
│   ├── streaming/         Kafka producers and live consumer
│   ├── bigdata/           Spark batch jobs
│   └── coach/             live feedback (voice)
├── scripts/analysis/      one-off analyses (dataset stats, baselines, camera angle)
├── notebooks/             exploration notebooks
├── experiments/           one folder per experiment: config + metrics
├── models/                trained checkpoints (not in git)
├── tests/                 sanity checks on data and code
├── reports/               briefing PDF, figures, final report
├── presentation/          validation web app (open index.html)
└── docs/                  course documents and notes
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .                # makes `import awc` work everywhere
python -m pytest                # data tests skip when data/raw is absent
```

Spark also needs **Java 17** (`JAVA_HOME`) and **Hadoop winutils** for Windows (`HADOOP_HOME=C:\hadoop`, with `bin\winutils.exe` and `bin\hadoop.dll`). Spark scratch files go to `C:\awc_tmp`, not into this OneDrive folder. Details are in `docs/decisions.md`.

Week 1 demo, from the project root with the venv active:

```bash
python -m awc.bigdata.build_interim --sessions w00 --cores local[2]
python -m awc.training.train_m1 --arch cnn --inputs both --quick
python scripts/eval_rep_counting.py
```

## Results so far

First measurements (quick tests, not final results):

- **Fatigue is visible**: in 45 of 64 MM-Fit squat sets, the last reps are slower than the first ones.
- **Rep counting from the skeleton** (early hip-height check): exact in 63/64 squat sets and 62/62 lunge sets.
- **Model 1** (step 1.5, the real benchmark): transformer on pose + IMU, macro-F1 0.880 and accuracy 0.950 on MM-Fit people never seen in training. Table in `reports/m1_benchmark.md`.
- **Rep counting** (exercise-specific signal, fixed 2026-10-24): macro MAE **0.49** reps per set across 616 sets (target 0.50). Lunges 0.02, squats 0.19, curls 0.46. Table in `reports/rep_counting.md`.
- **Form mistakes on an unseen person** (random forest, train on 3 people, test on the 4th): squat 53 %, lunge 82 %, plank 90 % (chance 20–33 %). EC3D authors: 90.9 %.
- **Camera angle**: squat mistakes are best recognized at about 45° in front (71 %), vs ~55 % from the front or the side.

Reproduce with the scripts in `scripts/analysis/`.

## Tools

All free: Python, PyTorch, MediaPipe, Apache Kafka, Apache Spark, pandas, scikit-learn.
