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
| 1.2 | Capture decision, recorder, recording protocol | ✅ Code done · Oct 22, 2026 · sessions not recorded yet |
| 1.3 | Spark sync of MM-Fit to Parquet | ✅ Done · Oct 22, 2026 |
| 1.4 | One skeleton for every source, Model 1 windows | ✅ Done · Oct 22, 2026 |
| 1.5 | Model 1 benchmark and rep counting | ✅ Done · Oct 22, 2026 |
| 1.6 | Week 1 validation package | ✅ Done · Oct 22, 2026 |
| 2.1 | EC3D rep dataset for Model 2 | ⏳ Next |
| 2.2–3.5 | Model 2, Kafka, Model 3, live coach, report | ⬜ |

### Log

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
| Our recordings | Fatigue over weeks | 2 people, 3–6 weeks (to record) |

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
