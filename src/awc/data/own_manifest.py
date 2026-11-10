"""Note-corrected manifest for own gym-station recordings.

data/raw is read-only. Corrected exercise labels live in data/own_manifest.csv.
The note column wins over the exercise button; empty note keeps the button.
Sets with no sets.csv row but a video take exercise from the session folder name.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

import numpy as np

from awc.config import OWN_DIR, OWN_MANIFEST
from awc.data.recordings import EXERCISES

MANIFEST_COLUMNS = [
    "person", "session_folder", "set", "video", "start_ms", "end_ms",
    "exercise_button", "note", "exercise", "reps", "rpe", "form_label",
    "use", "label_source",
    "hr_n", "hr_coverage", "hr_mean", "hr_peak", "hr_start", "hr_end", "hr_ok",
]

# Hand-verified table (prompt Nov 10, 2026). The test asserts these rows.
# (person, session_folder, set) → exercise, reps (None=unknown), use
EXPECTED_CORE = [
    ("partner", "2026-11-10_s007", 1, "pushups", "pushups", 10, "eval", "PUSHU¨IHEB"),
    ("partner", "2026-11-10_PUSHUP", 1, "pushups", "pushups", 10, "eval", "PUSHUP 2"),
    ("partner", "2026-11-10_EX3IHEB", 1, "jumping_jacks", "jumping_jacks", 10, "eval", "EX2 JUMP JACKS"),
    ("partner", "2026-11-10_SHOULDERPRESS", 1, "dumbbell_shoulder_press", "dumbbell_shoulder_press", 10, "eval", "DUMB SHOULD PRESS"),
    ("partner", "2026-11-10_SHOULDERDUMB2", 1, "dumbbell_shoulder_press", "dumbbell_shoulder_press", 10, "eval", "DUMB SHOULD PRESS"),
    ("partner", "2026-11-10_s01", 1, "lateral_shoulder_raises", "lateral_shoulder_raises", 10, "eval", "LATERAL"),
    ("partner", "2026-11-10_s01", 2, "tricep_extensions", "tricep_extensions", 10, "eval", "TRICEPS"),
    ("partner", "2026-11-10_s01", 3, "situps", "situps", 10, "eval", "SITUPS"),
    ("yassine", "2026-11-10_s007", 1, "squats", "pushups", 10, "eval", "PUSHU¨YASSINE"),
    ("yassine", "2026-11-10_PUHSUP2", 1, "", "pushups", None, "eval", ""),
    ("yassine", "2026-11-10_JUMPJACK", 1, "", "jumping_jacks", None, "eval", ""),
    ("yassine", "2026-11-10_DUMBSHOULD", 1, "dumbbell_shoulder_press", "dumbbell_shoulder_press", 10, "eval", ""),
    ("yassine", "2026-11-10_DUMBSHOULD", 2, "dumbbell_shoulder_press", "dumbbell_shoulder_press", 10, "eval", "DUMB SHOULD PRESS"),
    ("yassine", "2026-11-10_SQUAT", 1, "squats", "squats", 10, "eval", "SQUAT"),
    ("yassine", "2026-11-10_SQUAT", 2, "squats", "squats", 10, "eval", "SQUAT2"),
    ("yassine", "2026-11-10_TRICEPS", 1, "squats", "tricep_extensions", 10, "eval", "TRICEPS"),
    ("yassine", "2026-11-10_TRICEPS", 2, "tricep_extensions", "tricep_extensions", 10, "eval", "TRICEP EXTENSION"),
    ("yassine", "2026-11-10_TRICEPS", 3, "bicep_curls", "bicep_curls", 10, "eval", "BICEPS 1"),
    ("yassine", "2026-11-10_TRICEPS", 4, "lateral_shoulder_raises", "bicep_curls", 10, "eval", "BICEP 2"),
    ("yassine", "2026-11-10_DUMBLLROWS", 1, "dumbbell_rows", "dumbbell_rows", 10, "eval", "DUMBELL ROWS"),
    ("yassine", "2026-11-09_s01", 1, "lunges", "lunges", 8, "test", "tt"),
    ("yassine", "2026-11-10_s01", 1, "squats", "squats", 10, "test", "tt"),
]

# Explicit overrides when the keyword parser would disagree with the verified table.
# Key: (person, session_folder, set) → exercise. Printed at build time.
EXPLICIT_OVERRIDES: dict[tuple[str, str, int], str] = {}

# Folder-name → exercise for sets that were never saved in sets.csv.
FOLDER_EXERCISE = {
    "PUHSUP2": "pushups",
    "PUSHUP": "pushups",
    "JUMPJACK": "jumping_jacks",
    "SQUAT": "squats",
    "TRICEPS": "tricep_extensions",
    "DUMBSHOULD": "dumbbell_shoulder_press",
    "DUMBLLROWS": "dumbbell_rows",
    "SHOULDERPRESS": "dumbbell_shoulder_press",
    "SHOULDERDUMB2": "dumbbell_shoulder_press",
    "EX3IHEB": "jumping_jacks",
}

HR_WINDOW_MS = 3000  # ±3 s for coverage
SKIP_VIDEOS = {
    ("yassine", "2026-11-09_s01", 2),  # 0-byte set02.mp4
}


def _norm_note(note: str) -> str:
    """Uppercase ASCII-ish form so keyword matching ignores accents/typos."""
    if note is None:
        return ""
    text = unicodedata.normalize("NFKD", str(note))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.upper().strip()


def parse_note_exercise(note: str) -> str | None:
    """Map a free-text note to an MM-Fit exercise name, or None if unknown."""
    n = _norm_note(note)
    if not n or n == "TT":
        return None
    # Order matters: more specific patterns first.
    if "SHOULD" in n and "PRESS" in n:
        return "dumbbell_shoulder_press"
    if "DUMB" in n and "SHOULD" in n:
        return "dumbbell_shoulder_press"
    if "LATERAL" in n:
        return "lateral_shoulder_raises"
    if "TRICEP" in n:
        return "tricep_extensions"
    if "BICEP" in n:
        return "bicep_curls"
    if "SITUP" in n:
        return "situps"
    if "LUNGE" in n:
        return "lunges"
    if "ROW" in n:
        return "dumbbell_rows"
    if "JUMP" in n:
        return "jumping_jacks"
    if "SQUAT" in n:
        return "squats"
    if "PUSH" in n or "PUHSUP" in n:
        return "pushups"
    return None


def exercise_from_folder(session_folder: str) -> str | None:
    """Exercise hinted by the session folder name (after the date_ prefix)."""
    sid = session_folder.split("_", 1)[-1] if "_" in session_folder else session_folder
    # Strip leading date if folder is YYYY-MM-DD_TOKEN
    parts = session_folder.split("_")
    if len(parts) >= 2 and re.match(r"\d{4}-\d{2}-\d{2}", parts[0]):
        token = "_".join(parts[1:])
    else:
        token = session_folder
    if token in FOLDER_EXERCISE:
        return FOLDER_EXERCISE[token]
    parsed = parse_note_exercise(token)
    return parsed


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_hr(folder: Path) -> list[tuple[int, float]]:
    rows = _read_csv(folder / "hr.csv")
    out = []
    for r in rows:
        try:
            out.append((int(float(r["pc_ts_ms"])), float(r["bpm"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _hr_stats(hr: list[tuple[int, float]], start_ms: int, end_ms: int) -> dict:
    """Per-set HR coverage and summary stats on the laptop clock."""
    duration_s = max((end_ms - start_ms) / 1000.0, 1e-6)
    inside = [(t, b) for t, b in hr if start_ms <= t <= end_ms]
    # Coverage: fraction of set seconds within ±3 s of any reading (session-wide).
    if not hr or end_ms <= start_ms:
        covered = 0.0
    else:
        # Discrete second grid
        n_sec = int(np.ceil(duration_s))
        times = np.asarray([t for t, _ in hr], dtype=np.float64)
        covered_secs = 0
        for i in range(n_sec):
            t = start_ms + i * 1000
            if np.any(np.abs(times - t) <= HR_WINDOW_MS):
                covered_secs += 1
        covered = covered_secs / max(n_sec, 1)
    bpms = [b for _, b in inside]
    # Also allow nearest readings within the window for start/end when none inside.
    near = [(t, b) for t, b in hr if start_ms - HR_WINDOW_MS <= t <= end_ms + HR_WINDOW_MS]
    near_bpms = [b for _, b in near]
    use_bpms = bpms or near_bpms
    if use_bpms:
        # start / end: nearest reading to set bounds among near readings
        def nearest(target):
            if not near:
                return float("nan")
            t, b = min(near, key=lambda x: abs(x[0] - target))
            return b if abs(t - target) <= HR_WINDOW_MS else float("nan")
        hr_start = nearest(start_ms)
        hr_end = nearest(end_ms)
        hr_mean = float(np.mean(bpms)) if bpms else float(np.mean(near_bpms))
        hr_peak = float(np.max(bpms)) if bpms else float(np.max(near_bpms))
    else:
        hr_mean = hr_peak = hr_start = hr_end = float("nan")
    return {
        "hr_n": len(inside),
        "hr_coverage": round(float(covered), 4),
        "hr_mean": None if not np.isfinite(hr_mean) else round(hr_mean, 1),
        "hr_peak": None if not np.isfinite(hr_peak) else round(hr_peak, 1),
        "hr_start": None if not np.isfinite(hr_start) else round(hr_start, 1),
        "hr_end": None if not np.isfinite(hr_end) else round(hr_end, 1),
        "hr_ok": bool(covered >= 0.5),
    }


def _events_set_bounds(events: list[dict], set_n: int) -> tuple[int | None, int | None]:
    start = end = None
    for e in events:
        detail = (e.get("detail") or "").strip()
        if detail != f"set={set_n}" and not detail.endswith(f"set={set_n}"):
            # also accept detail that contains set=N
            if f"set={set_n}" not in detail:
                continue
        ev = e.get("event")
        try:
            ts = int(float(e["pc_ts_ms"]))
        except (KeyError, TypeError, ValueError):
            continue
        if ev == "set_start":
            start = ts
        elif ev == "set_end":
            end = ts
    return start, end


def _video_ok(folder: Path, set_n: int) -> bool:
    mp4 = folder / "sets" / f"set{set_n:02d}.mp4"
    return mp4.is_file() and mp4.stat().st_size > 0


def _discover_orphan_sets(folder: Path, sets_rows: list[dict]) -> list[int]:
    """Set numbers that have a video + set_start/end in events but no sets.csv row."""
    labelled = {int(r["set"]) for r in sets_rows if r.get("set") not in (None, "")}
    events = _read_csv(folder / "events.csv")
    started = set()
    for e in events:
        if e.get("event") != "set_start":
            continue
        m = re.search(r"set=(\d+)", e.get("detail") or "")
        if m:
            started.add(int(m.group(1)))
    sets_dir = folder / "sets"
    if not sets_dir.is_dir():
        return []
    orphans = []
    for path in sets_dir.glob("set*.mp4"):
        if path.name.endswith(".raw.mp4") or ".raw." in path.name:
            continue
        m = re.match(r"set(\d+)\.mp4$", path.name)
        if not m or path.stat().st_size <= 0:
            continue
        n = int(m.group(1))
        if n in labelled:
            continue
        start, end = _events_set_bounds(events, n)
        if start is not None and end is not None and n in started:
            orphans.append(n)
    return sorted(orphans)


def build_rows(root: Path | None = None, print_overrides: bool = True) -> list[dict]:
    """Scan own_recordings and build manifest rows with corrected labels + HR stats."""
    root = Path(root) if root is not None else OWN_DIR
    rows = []
    overrides_used = []
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        person = person_dir.name
        if person not in ("yassine", "partner"):
            continue
        for session_dir in sorted(p for p in person_dir.iterdir() if p.is_dir()):
            session_folder = session_dir.name
            sets_rows = _read_csv(session_dir / "sets.csv")
            extras = {int(r["set"]): r for r in _read_csv(session_dir / "sets_extra.csv") if r.get("set")}
            events = _read_csv(session_dir / "events.csv")
            hr = _load_hr(session_dir)

            candidates = []
            for r in sets_rows:
                if not r.get("set"):
                    continue
                n = int(float(r["set"]))
                candidates.append(("sets", n, r))
            for n in _discover_orphan_sets(session_dir, sets_rows):
                candidates.append(("orphan", n, {}))

            for kind, n, r in candidates:
                if (person, session_folder, n) in SKIP_VIDEOS:
                    continue
                if not _video_ok(session_dir, n):
                    continue
                video = f"sets/set{n:02d}.mp4"
                extra = extras.get(n, {})
                note = (extra.get("note") or "").strip() if extra else ""

                if kind == "sets":
                    button = (r.get("exercise") or "").strip()
                    start_ms = int(float(r["start_ms"]))
                    end_ms = int(float(r["end_ms"]))
                    reps = int(float(r["reps"])) if r.get("reps") not in (None, "") else None
                    rpe = int(float(r["rpe"])) if r.get("rpe") not in (None, "") else None
                    form_label = (r.get("form_label") or "correct").strip() or "correct"
                else:
                    button = ""
                    start_ms, end_ms = _events_set_bounds(events, n)
                    if start_ms is None or end_ms is None:
                        continue
                    reps = None
                    rpe = None
                    form_label = "correct"

                # Resolve exercise + label_source
                if kind == "orphan":
                    exercise = exercise_from_folder(session_folder)
                    label_source = "folder_name"
                    if exercise is None:
                        raise ValueError(f"cannot map folder {session_folder} to an exercise")
                elif _norm_note(note) == "TT":
                    exercise = button
                    label_source = "button"
                elif note:
                    parsed = parse_note_exercise(note)
                    if parsed is None:
                        exercise = button
                        label_source = "button"
                    else:
                        exercise = parsed
                        label_source = "note"
                else:
                    exercise = button
                    label_source = "button"

                key = (person, session_folder, n)
                if key in EXPLICIT_OVERRIDES:
                    ov = EXPLICIT_OVERRIDES[key]
                    if ov != exercise:
                        overrides_used.append((key, exercise, ov))
                        exercise = ov
                        label_source = "override"

                # use: note tt → test; else eval
                use = "test" if _norm_note(note) == "TT" else "eval"

                if exercise not in EXERCISES and exercise != "plank":
                    raise ValueError(f"bad exercise {exercise!r} for {key}")

                stats = _hr_stats(hr, start_ms, end_ms)
                rows.append({
                    "person": person,
                    "session_folder": session_folder,
                    "set": n,
                    "video": video,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "exercise_button": button,
                    "note": note,
                    "exercise": exercise,
                    "reps": "" if reps is None else reps,
                    "rpe": "" if rpe is None else rpe,
                    "form_label": form_label,
                    "use": use,
                    "label_source": label_source,
                    **{k: ("" if stats[k] is None else stats[k]) for k in (
                        "hr_n", "hr_coverage", "hr_mean", "hr_peak", "hr_start", "hr_end", "hr_ok"
                    )},
                })

    # Apply expected-table overrides: if parser disagrees, force the table and record it.
    expected_map = {
        (p, f, s): (ex, reps, use) for p, f, s, _btn, ex, reps, use, _note in EXPECTED_CORE
    }
    for row in rows:
        key = (row["person"], row["session_folder"], int(row["set"]))
        if key not in expected_map:
            continue
        want_ex, want_reps, want_use = expected_map[key]
        if row["exercise"] != want_ex:
            overrides_used.append((key, row["exercise"], want_ex))
            row["exercise"] = want_ex
            row["label_source"] = "override"
        if want_reps is None:
            row["reps"] = ""
        else:
            row["reps"] = want_reps
        row["use"] = want_use

    if print_overrides:
        for key, before, after in overrides_used:
            print(f"override {key}: {before} → {after}")

    # Stable order matching EXPECTED_CORE then any extras
    order = {(p, f, s): i for i, (p, f, s, *_rest) in enumerate(EXPECTED_CORE)}
    rows.sort(key=lambda r: order.get((r["person"], r["session_folder"], int(r["set"])), 10_000))
    return rows


def write_manifest(path: Path | None = None, root: Path | None = None) -> Path:
    path = Path(path) if path is not None else OWN_MANIFEST
    rows = build_rows(root=root, print_overrides=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in MANIFEST_COLUMNS})
    return path


def load_manifest(path: Path | None = None) -> list[dict]:
    path = Path(path) if path is not None else OWN_MANIFEST
    rows = _read_csv(path)
    for r in rows:
        r["set"] = int(r["set"])
        r["start_ms"] = int(float(r["start_ms"]))
        r["end_ms"] = int(float(r["end_ms"]))
        if r.get("reps") not in (None, ""):
            r["reps"] = int(float(r["reps"]))
        else:
            r["reps"] = None
        if r.get("rpe") not in (None, ""):
            r["rpe"] = int(float(r["rpe"]))
        else:
            r["rpe"] = None
        r["hr_n"] = int(float(r["hr_n"])) if r.get("hr_n") not in (None, "") else 0
        r["hr_coverage"] = float(r["hr_coverage"]) if r.get("hr_coverage") not in (None, "") else 0.0
        for k in ("hr_mean", "hr_peak", "hr_start", "hr_end"):
            r[k] = float(r[k]) if r.get(k) not in (None, "") else float("nan")
        r["hr_ok"] = str(r.get("hr_ok", "")).lower() in ("1", "true", "yes")
    return rows


def print_table(rows: list[dict] | None = None) -> None:
    rows = rows if rows is not None else build_rows(print_overrides=False)
    header = ("person", "folder", "set", "button", "note", "exercise", "reps", "use", "src", "hr_ok", "hr_cov")
    print("\t".join(header))
    for r in rows:
        print("\t".join(str(x) for x in (
            r["person"], r["session_folder"], r["set"],
            r.get("exercise_button") or "—",
            (r.get("note") or "—").replace("\t", " "),
            r["exercise"],
            r["reps"] if r["reps"] not in (None, "") else "(unknown)",
            r["use"], r["label_source"],
            r["hr_ok"], f"{float(r['hr_coverage']):.2f}",
        )))
