"""Our own recordings: phone video, PC timestamps, watch heart rate, set labels.

A session lives in data/raw/own_recordings/<user>/<YYYY-MM-DD>_<session-id>/, in one of two layouts:
- PC recorder (scripts/record_session.py): video.mp4 + frames.csv for the whole session.
- Phone app (docs/gym/, imported by scripts/import_phone_session.py): sets/setNN.mp4 +
  frames_setNN.csv, one video per set; meta.json has "source": "phone-app".
Both have hr.csv, sets.csv and meta.json. Every timestamp of one session is on one clock.
"""
from datetime import date
from pathlib import Path

from awc.config import MMFIT_CLASSES, OWN_DIR

USERS = ("yassine", "partner")
EXERCISES = tuple(c for c in MMFIT_CLASSES if c != "non_activity")
FORM_LABELS = (
    "correct",
    "feet too wide",
    "knees inward",
    "not low enough",
    "front bent",
    "knee passes toe",
    "arched back",
    "hunched back",
)

FRAMES_HEADER = ["frame", "pc_ts_ms"]
HR_HEADER = ["pc_ts_ms", "bpm"]
SETS_HEADER = ["set", "exercise", "start_ms", "end_ms", "reps", "rpe", "form_label"]


def session_dir(user, session_id, on_date=None) -> Path:
    """Folder for one recording. on_date defaults to today."""
    if user not in USERS:
        raise ValueError(f"user must be one of {USERS}")
    if not session_id or any(c in session_id for c in " /\\"):
        raise ValueError("session-id must be a short token such as s01")
    on_date = on_date or date.today()
    return OWN_DIR / user / f"{on_date.isoformat()}_{session_id}"


def _write_header(path, header):
    path.write_text(",".join(header) + "\n", encoding="utf-8")


def init_session_files(folder):
    """Create the session folder and the three CSV files with their headers."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    _write_header(folder / "frames.csv", FRAMES_HEADER)
    _write_header(folder / "hr.csv", HR_HEADER)
    _write_header(folder / "sets.csv", SETS_HEADER)
    return folder


def validate_set(exercise, reps, rpe, form_label):
    """Raise ValueError when a set row would be illegal."""
    if exercise not in EXERCISES:
        raise ValueError(f"unknown exercise {exercise!r}")
    reps = int(reps)
    rpe = int(rpe)
    if reps < 1:
        raise ValueError("reps must be >= 1")
    if not 1 <= rpe <= 10:
        raise ValueError("RPE must be an integer from 1 to 10")
    if form_label not in FORM_LABELS:
        raise ValueError(f"unknown form label {form_label!r}")
    return exercise, reps, rpe, form_label


def append_set(folder, set_index, exercise, start_ms, end_ms, reps, rpe, form_label):
    exercise, reps, rpe, form_label = validate_set(exercise, reps, rpe, form_label)
    if int(end_ms) < int(start_ms):
        raise ValueError("end_ms is before start_ms")
    path = Path(folder) / "sets.csv"
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{int(set_index)},{exercise},{int(start_ms)},{int(end_ms)},{reps},{rpe},{form_label}\n")


def append_frame(path, frame, pc_ts_ms):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{int(frame)},{int(pc_ts_ms)}\n")


def append_hr(path, pc_ts_ms, bpm):
    bpm = float(bpm)
    if not 30 <= bpm <= 230:
        raise ValueError("bpm outside 30–230")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{int(pc_ts_ms)},{bpm:.1f}\n")


def write_meta(folder, meta: dict):
    import json
    (Path(folder) / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load(session_id):
    raise NotImplementedError("Own-session loading is step 2.4, after recordings exist.")
