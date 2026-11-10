/* Week 2 progress numbers — sourced from reports (rq4_own, week2, m2_benchmark) and experiments.
   Rebuild the HTML with: python scripts/docs/build_week2.py
*/
const WEEK2 = {
  date: "10 Nov 2026",
  team: ["Yassine Nemri", "Iheb Lamouchi"],
  course: "Deep Learning & Big Data · Validation 2 · week 2 progress",

  plan_checklist: [
    {step: "2.1", label: "EC3D rep dataset (LOSO, T=64)", status: "ok"},
    {step: "2.2", label: "Model 2 benchmark (ST-GCN / LSTM / RF)", status: "warn", note: "ST-GCN squat lead within fold noise"},
    {step: "2.3", label: "Kafka pose + HR streaming", status: "warn", note: "replay OK; live station Kafka not shown (Docker down)"},
    {step: "2.4", label: "RQ4 on MM-Fit video + our videos", status: "ok"},
    {step: "2.5", label: "Spark rep table + within-session baselines", status: "ok"},
    {step: "2.5b", label: "Spark own rep table", status: "ok"},
    {step: "2.6", label: "Week 2 validation package", status: "ok"},
    {step: "2.6b", label: "Station post-set live detection", status: "ok"},
  ],

  own: {
    people: 2,
    eval_sets: 20,
    test_sets: 2,
    exercises_eval: 9,
    exercises_total_mmfit: 10,
    yassine_eval: 12,
    partner_eval: 8,
    fps: 30.0,
    sets_uploaded_ok: 22,
    wrong_button_corrections: 3,
    hr_yassine: {n: 12, hr_ok: 7, mean_cov: 0.54, median_cov: 0.58},
    hr_partner: {n: 8, hr_ok: 4, mean_cov: 0.52, median_cov: 0.59},
    rtt_ms: {yassine_s01: 3.5, partner_s007: 3.5, yassine_s09: 7.6},
    reps_verified: false,
    reps_typed_provisional: {mae: 3.17, exact_pct: 22.2, within1_pct: 50.0, n: 18},
  },

  m2_ec3d: {
    stgcn_squat: "55.6 ± 12.4",
    rf_squat: "53.8 ± 6.1",
    stgcn_lunge: "68.5 ± 8.2",
    rf_lunge: "71.6 ± 19.3",
    stgcn_plank: "70.2 ± 19.1",
    rf_plank: "82.8 ± 7.3",
    lstm_squat: "33.3 ± 11.5",
    folds: 4,
    conclusion: "ST-GCN squat lead over RF is within fold noise. RF wins on lunge and plank. Do not treat ST-GCN as the overall winner.",
  },

  kafka: {
    pose_msgs: 29.8,
    pose_lat_ms: 48,
    hr_msgs: 0.79,
    hr_lat_ms: 19,
    mode: "replay only",
    docker_live: false,
    note: "Docker daemon not running on Nov 10 — station --kafka not measured.",
  },

  rq4_own: {
    set_acc: 0.70,
    set_correct: 14,
    n: 20,
    wilson: [0.48, 0.85],
    macro_f1: 0.511,
    window_acc: 0.629,
    n_windows: 299,
    n_no_detection: 2,
    yassine: {n: 12, acc: 0.750},
    partner: {n: 8, acc: 0.625},
    perfect: [
      {ex: "bicep_curls", n: 2, acc: 1.0},
      {ex: "dumbbell_shoulder_press", n: 4, acc: 1.0},
      {ex: "jumping_jacks", n: 2, acc: 1.0},
      {ex: "pushups", n: 4, acc: 1.0},
      {ex: "squats", n: 2, acc: 1.0},
    ],
    failing: [
      {ex: "dumbbell_rows", n: 1, acc: 0.0, pred: "pushups"},
      {ex: "lateral_shoulder_raises", n: 1, acc: 0.0, pred: "jumping_jacks"},
      {ex: "situps", n: 1, acc: 0.0, pred: "pushups"},
      {ex: "tricep_extensions", n: 3, acc: 0.0, pred: "no_exercise_detected / rows"},
    ],
    compare: [
      {src: "MM-Fit test_unseen (dataset pose)", metric: "macro-F1", value: 0.85},
      {src: "MM-Fit w20 MediaPipe", metric: "macro-F1", value: 0.80},
      {src: "RepCount public videos", metric: "video acc (4 ex.)", value: 0.808},
      {src: "Our videos", metric: "set accuracy", value: 0.70, n: 20},
      {src: "Our videos", metric: "macro-F1 (9 ex.)", value: 0.511, n: 20},
    ],
  },

  diagnosis: [
    {ex: "tricep_extensions", pred: "no_exercise_detected", line: "MM-Fit is overhead standing; ours stay low-amplitude near the hip — likely seated / kickback."},
    {ex: "lateral_shoulder_raises", pred: "jumping_jacks", line: "Arm arcs match; MM-Fit jacks add ankle bounce — model still keys on arms."},
    {ex: "situps", pred: "pushups", line: "Our ankles stay ~−1.2 below mid-hip (legs on floor); MM-Fit sit-ups put ankles near the hip line."},
    {ex: "dumbbell_rows", pred: "pushups", line: "Ours are one-sided and body-horizontal from the side — low wrist looks like a push-up support."},
  ],

  lopo: {
    note: "Separate experiment. Zero-shot remains the main RQ4 result.",
    directions: [
      {adapt: "yassine", test: "partner", n: 8, zero: 0.625, adapted: 0.625},
      {adapt: "partner", test: "yassine", n: 12, zero: 0.75, adapted: 0.917},
    ],
  },

  spark_own: {
    rep_rows: 269,
    eval_sets: 20,
    test_sets: 2,
    seg_failed: 4,
    velocity_loss: {squats: 0.222, pushups: 0.065, tricep_extensions: -1.146},
  },

  live: {
    cached_old: "0.84 / 0.03 / 0.03 s",
    full_mean_s: 13.11,
    full_pose_s: 13.10,
    live_fast_mean_s: 7.80,
    live_fast_pose_s: 7.79,
    n_sets: 3,
    exercise_match: "3/3",
    form_p_correct: 0.06,
    form_n_reps: 20,
    form_note: "hidden until calibrated (P(correct) 0.06 on our correct squats)",
  },

  limits: [
    "Small n: 20 eval sets, one day, two people.",
    "No lunges in eval (only a poorly framed test set).",
    "HR gaps: watch silent 20–80 s; coverage ≈ 0.5 on eval.",
    "M2 does not transfer to our camera (P(correct) 0.06).",
    "RPE not usable (station defaulted to 10).",
    "Rep MAE still provisional until hand-verified counts exist.",
  ],

  next: [
    "Longer fatigue sessions (same person, multi-set).",
    "Model 3 on our data with personal baselines.",
    "Calibrate form (or keep it hidden).",
    "Record lunges; more people.",
  ],
};
