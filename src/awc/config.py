"""Project paths and constants. Every script and module imports paths from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
MMFIT_DIR = RAW / "mm-fit"              # w00 … w20, one folder per workout session
EC3D_DIR = RAW / "ec3d"                 # data_3D.pickle, data.pickle
OWN_DIR = RAW / "own_recordings"        # our phone + Galaxy Watch 7 sessions (real only)
MMFIT_VIDEO_DIR = RAW / "mm-fit-video"  # RGB videos for replay / sync checks
REPCOUNT_DIR = RAW / "repcount"         # RepCount-A test split (Lance), public gym videos with rep counts
INTERIM = DATA / "interim"              # synced, cleaned, normalized (Parquet)
REPLAY_DIR = INTERIM / "replay"         # Kafka sink output for replay sessions (never own_recordings)
MMFIT_VIDEO_INTERIM = INTERIM / "mmfit_video"  # MediaPipe pose from raw MM-Fit RGB, sensor clock
VIDEO_SYNC_PATH = INTERIM / "video_sync.json"  # per-session video↔sensor linear fits
PROCESSED = DATA / "processed"          # model-ready windows and per-rep tensors
M2_REPS = PROCESSED / "m2_reps.npz"     # EC3D reps for Model 2 (3D sequences; angles applied later)

# Default replay sources for Step 2.3 streaming tests (not for fatigue / model scores).
REPLAY_POSE_VIDEO = MMFIT_VIDEO_DIR / "w20_rgb.mp4"
REPLAY_HR_NPY = MMFIT_DIR / "w20" / "w20_sw_l_hr.npy"

# Approximate w20 video↔sensor drift (measured on w20 only; 8/10 sampled sets within 3 frames).
# sensor_frame ≈ video_frame + 3 + 0.00169 * (video_frame - 2178). Sit-up and tricep samples did not fit.
# Use for streaming replay timing only. Do not treat as a calibrated sync for model results.
W20_VIDEO_SENSOR_OFFSET = 3.0
W20_VIDEO_SENSOR_SLOPE = 0.00169
W20_VIDEO_SENSOR_PIVOT = 2178.0

MODELS = ROOT / "models"                # trained checkpoints
POSE_MODEL = MODELS / "pose_landmarker_full.task"   # MediaPipe Pose Landmarker (pretrained, downloaded)
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
EXPERIMENTS = ROOT / "experiments"      # one folder per experiment: config + metrics
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
PRESENTATION = ROOT / "presentation"

VIDEO_FPS = 30                          # MM-Fit and EC3D video clock; every stream is aligned to it
WINDOW_SECONDS = 3                      # Model 1 input window
WINDOW_OVERLAP = 0.5

# Spark and Kafka scratch space lives outside OneDrive so sync cannot lock files.
TMP = Path("C:/awc_tmp")
SPARK_LOCAL_DIR = TMP / "spark"
SPARK_WAREHOUSE = TMP / "spark-warehouse"
KAFKA_TMP = TMP / "kafka"
KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPICS = ("pose.keypoints", "watch.hr", "watch.imu", "coach.events")
HADOOP_HOME = Path("C:/hadoop")

# Official MM-Fit split from the authors' train_multimodal_ar.py.
# test_unseen is the main RQ1 test (people never seen in training).
MMFIT_SPLIT = {
    "train": ["w01", "w02", "w03", "w04", "w06", "w07", "w08", "w16", "w17", "w18"],
    "val": ["w14", "w15", "w19"],
    "test_seen": ["w09", "w10", "w11"],
    "test_unseen": ["w00", "w05", "w12", "w13", "w20"],
}
MMFIT_CLASSES = [
    "squats", "lunges", "bicep_curls", "situps", "pushups", "tricep_extensions",
    "dumbbell_rows", "jumping_jacks", "dumbbell_shoulder_press",
    "lateral_shoulder_raises", "non_activity",
]
