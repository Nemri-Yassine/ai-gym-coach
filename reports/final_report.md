# AI Workout Coach · Final report (short)

**Course:** Deep Learning & Big Data · 2026–2027  
**Team:** Yassine Nemri, Iheb Lamouchi  
**Date:** 7 Nov 2026

## Objective

Build a coach that recognizes the exercise, counts reps, checks form, and warns before fatigue becomes risky — using a phone camera and a smartwatch, with a **personal baseline** as the main contribution.

## Research questions

1. **RQ1:** Public-data models on unseen people?  
2. **RQ2:** Does heart rate beat movement alone for fatigue risk?  
3. **RQ3:** Does a personal baseline warn earlier than a general model?  
4. **RQ4:** Do the models still work on raw RGB (phone-like) video?

## Data

- **MM-Fit:** 21 sessions, pose + watch IMU + HR (HR missing in w05/w10/w18).  
- **EC3D:** form mistakes for Model 2.  
- **Own recordings:** none (injury). Replaced by MM-Fit RGB (Zenodo 7672767) + within-session baselines. No RPE.

## Approach

Kafka (replay) → MediaPipe / dataset pose → Model 1 (exercise) → rep counter → Model 2 (form RF) → Spark rep history → Model 3 (GRU fatigue/risk, B1–B5) → coach policy (+ optional voice).

## Results

| RQ | Finding |
|---|---|
| RQ1 | Transformer pose+IMU macro-F1 **0.880** unseen; rep MAE **0.49** |
| RQ2 | B4 AUC **0.803** vs B3 **0.814** — HR does **not** help here |
| RQ3 | B2 (general) **0.884** > B4 (personal) **0.803**; B4 ≫ B1 rule (0.50) |
| RQ4 | w20 MediaPipe: M1 0.85→0.80; rep MAE 0.93→1.56 |
| Spark | 5,918 rep rows; within-session baselines |
| Kafka | pose 29.8 msg/s @ 48 ms; HR 0.79 @ 19 ms |

Full tables: `reports/results.md`, `reports/m3_results.md`, `reports/week2.md`.

## Limits

- No own recordings (injury); no multi-week personal history  
- No RPE; Spearman uses `set_idx_in_exercise` instead  
- Within-session baseline only  
- Breakdown labels share the speed signal with the inputs (and with B1)  
- Live Galaxy Watch link not tested  
- MM-Fit rep counts assumed correct (teacher’s question — open limitation)  
- RQ4 evaluated on **w20 only** (other videos still downloading)  
- Coach demo: replay / text-first (`--no-voice` available)
