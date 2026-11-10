"""Our own recordings: phone video, PC timestamps, watch heart rate, set labels.

A session lives in data/raw/own_recordings/<user>/<YYYY-MM-DD>_<session-id>/, in one of two layouts:
- PC recorder (scripts/record_session.py): video.mp4 + frames.csv for the whole session.
- Phone / gym-station (docs/gym/, awc.station): sets/setNN.mp4 + frames_setNN.csv, one video
  per set; meta.json has "source": "phone-app" or "gym-station-iphone".
Both have hr.csv, sets.csv and meta.json. Every timestamp of one session is on one clock.
"""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from awc.config import MMFIT_CLASSES, OWN_DIR

USERS = ("yassine", "partner")
# Recording vocabulary (includes plank for Model 2 form labels). Model 1 classes stay MMFIT_CLASSES.
EXERCISES = tuple(c for c in MMFIT_CLASSES if c != "non_activity") + ("plank",)
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
FORM_BY_EXERCISE = {
    "squats": ("correct", "feet too wide", "knees inward", "not low enough", "front bent"),
    "lunges": ("correct", "not low enough", "knee passes toe"),
    "plank": ("correct", "arched back", "hunched back"),
}

FRAMES_HEADER = ["frame", "pc_ts_ms"]
HR_HEADER = ["pc_ts_ms", "bpm"]
HR_RR_HEADER = ["pc_ts_ms", "rr_ms"]
SETS_HEADER = ["set", "exercise", "start_ms", "end_ms", "reps", "rpe", "form_label"]
SETS_EXTRA_HEADER = [
    "set", "session_type", "weight_kg", "hold_s", "note",
    "hr_mean", "hr_peak", "hr_start", "hr_end", "duration_s",
]
EVENTS_HEADER = ["pc_ts_ms", "event", "detail"]
CLOCK_SYNC_HEADER = ["pc_ts_ms", "offset_ms", "rtt_ms"]


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


def init_session_files(folder, station=False):
    """Create the session folder and CSV headers. station=True adds extras / events / clock files."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "sets").mkdir(exist_ok=True)
    frames = folder / "frames.csv"
    hr = folder / "hr.csv"
    sets = folder / "sets.csv"
    if not frames.exists():
        _write_header(frames, FRAMES_HEADER)
    if not hr.exists():
        _write_header(hr, HR_HEADER)
    if not sets.exists():
        _write_header(sets, SETS_HEADER)
    if station:
        for name, header in (
            ("sets_extra.csv", SETS_EXTRA_HEADER),
            ("events.csv", EVENTS_HEADER),
            ("hr_rr.csv", HR_RR_HEADER),
            ("clock_sync.csv", CLOCK_SYNC_HEADER),
        ):
            path = folder / name
            if not path.exists():
                _write_header(path, header)
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
    allowed = FORM_BY_EXERCISE.get(exercise)
    if allowed is not None and form_label not in allowed:
        raise ValueError(f"form label {form_label!r} not valid for {exercise}")
    return exercise, reps, rpe, form_label


def append_set(folder, set_index, exercise, start_ms, end_ms, reps, rpe, form_label):
    exercise, reps, rpe, form_label = validate_set(exercise, reps, rpe, form_label)
    if int(end_ms) < int(start_ms):
        raise ValueError("end_ms is before start_ms")
    path = Path(folder) / "sets.csv"
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{int(set_index)},{exercise},{int(start_ms)},{int(end_ms)},{reps},{rpe},{form_label}\n")


def append_set_extra(folder, set_index, **fields):
    path = Path(folder) / "sets_extra.csv"
    if not path.exists():
        _write_header(path, SETS_EXTRA_HEADER)
    row = {h: "" for h in SETS_EXTRA_HEADER}
    row["set"] = int(set_index)
    for key, value in fields.items():
        if key in row and value is not None:
            row[key] = value
    with path.open("a", encoding="utf-8") as f:
        f.write(",".join(str(row[h]) for h in SETS_EXTRA_HEADER) + "\n")


def append_frame(path, frame, pc_ts_ms):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{int(frame)},{int(pc_ts_ms)}\n")


def append_hr(path, pc_ts_ms, bpm):
    bpm = float(bpm)
    if not 30 <= bpm <= 230:
        raise ValueError("bpm outside 30–230")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{int(pc_ts_ms)},{bpm:.1f}\n")


def append_event(folder, pc_ts_ms, event, detail=""):
    path = Path(folder) / "events.csv"
    if not path.exists():
        _write_header(path, EVENTS_HEADER)
    detail = str(detail).replace(",", ";").replace("\n", " ")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{int(pc_ts_ms)},{event},{detail}\n")


def write_meta(folder, meta: dict):
    (Path(folder) / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _read_csv(path):
    if not Path(path).is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def next_set_index(folder) -> int:
    """Next set number for resume (1-based). Never overwrites existing set videos."""
    rows = _read_csv(Path(folder) / "sets.csv")
    if not rows:
        return 1
    return max(int(r["set"]) for r in rows) + 1


def _events_bounds(events, set_n):
    start = end = None
    needle = f"set={set_n}"
    for e in events:
        detail = e.get("detail") or ""
        if needle not in detail:
            continue
        try:
            ts = int(float(e["pc_ts_ms"]))
        except (KeyError, TypeError, ValueError):
            continue
        if e.get("event") == "set_start":
            start = ts
        elif e.get("event") == "set_end":
            end = ts
    return start, end


def load(session_folder):
    """Load a per-set own recording (phone-app or gym-station).

    Returns a dict with sets, extras, frames (per set), hr, events, meta.
    Sets that have a video + set_start/set_end in events.csv but no sets.csv row
    are included with empty labels (exercise/reps unknown) so callers can still
    find the video paths.
    """
    folder = Path(session_folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    sets = _read_csv(folder / "sets.csv")
    for row in sets:
        for key in ("set", "start_ms", "end_ms", "reps", "rpe"):
            if row.get(key) not in (None, ""):
                row[key] = int(float(row[key]))
    extras = {int(r["set"]): r for r in _read_csv(folder / "sets_extra.csv") if r.get("set")}
    events = _read_csv(folder / "events.csv")
    labelled = {int(r["set"]) for r in sets}
    sets_dir = folder / "sets"
    if sets_dir.is_dir():
        for path in sorted(sets_dir.glob("set*.mp4")):
            if ".raw." in path.name or path.stat().st_size <= 0:
                continue
            try:
                n = int(path.stem.replace("set", ""))
            except ValueError:
                continue
            if n in labelled:
                continue
            start, end = _events_bounds(events, n)
            if start is None or end is None:
                continue
            sets.append({
                "set": n,
                "exercise": "",
                "start_ms": start,
                "end_ms": end,
                "reps": None,
                "rpe": None,
                "form_label": "correct",
            })
            labelled.add(n)
    frames = {}
    for row in sets:
        n = int(row["set"])
        path = folder / f"frames_set{n:02d}.csv"
        rows = _read_csv(path)
        frames[n] = [(int(r["frame"]), int(float(r["pc_ts_ms"]))) for r in rows]
        video_mp4 = folder / "sets" / f"set{n:02d}.mp4"
        video_webm = folder / "sets" / f"set{n:02d}.webm"
        row["video"] = str(video_mp4 if video_mp4.is_file() and video_mp4.stat().st_size > 0
                           else video_webm if video_webm.is_file() and video_webm.stat().st_size > 0
                           else "")
        row["extra"] = extras.get(n, {})
    hr = [(int(float(r["pc_ts_ms"])), float(r["bpm"])) for r in _read_csv(folder / "hr.csv")]
    clock = _read_csv(folder / "clock_sync.csv")
    return {
        "folder": folder,
        "meta": meta,
        "sets": sets,
        "frames": frames,
        "hr": hr,
        "events": events,
        "clock_sync": clock,
    }
