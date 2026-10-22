"""Project paths and constants. Every script and module imports paths from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
MMFIT_DIR = RAW / "mm-fit"              # w00 … w20, one folder per workout session
EC3D_DIR = RAW / "ec3d"                 # data_3D.pickle, data.pickle
OWN_DIR = RAW / "own_recordings"        # our phone + Galaxy Watch 7 sessions
INTERIM = DATA / "interim"              # synced, cleaned, normalized (Parquet)
PROCESSED = DATA / "processed"          # model-ready windows and per-rep tensors

MODELS = ROOT / "models"                # trained checkpoints
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
