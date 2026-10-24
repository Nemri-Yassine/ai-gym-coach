"""Recorder file format. No camera and no watch."""
from datetime import date

import pytest

from awc.config import OWN_DIR
from awc.data.recordings import (
    FRAMES_HEADER,
    HR_HEADER,
    SETS_HEADER,
    append_set,
    init_session_files,
    session_dir,
    validate_set,
)


def test_folder_naming():
    path = session_dir("yassine", "s01", date(2026, 10, 22))
    assert path == OWN_DIR / "yassine" / "2026-10-22_s01"
    with pytest.raises(ValueError):
        session_dir("someone", "s01", date(2026, 10, 22))


def test_csv_headers_and_set_validation(tmp_path):
    init_session_files(tmp_path)
    assert (tmp_path / "frames.csv").read_text(encoding="utf-8").strip() == ",".join(FRAMES_HEADER)
    assert (tmp_path / "hr.csv").read_text(encoding="utf-8").strip() == ",".join(HR_HEADER)
    assert (tmp_path / "sets.csv").read_text(encoding="utf-8").strip() == ",".join(SETS_HEADER)
    validate_set("squats", 8, 9, "correct")
    validate_set("lunges", 8, 1, "knee passes toe")
    with pytest.raises(ValueError):
        validate_set("squats", 8, 11, "correct")
    with pytest.raises(ValueError):
        validate_set("dance", 8, 5, "correct")
    with pytest.raises(ValueError):
        validate_set("squats", 8, 5, "invented mistake")
    append_set(tmp_path, 1, "squats", 1000, 5000, 8, 9, "correct")
    row = (tmp_path / "sets.csv").read_text(encoding="utf-8").strip().splitlines()[1]
    assert row == "1,squats,1000,5000,8,9,correct"


# ---- recorder loop, with a fake 20 fps camera, scripted keys and scripted answers ----
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_recorder():
    spec = importlib.util.spec_from_file_location("record_session", ROOT / "scripts" / "record_session.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_estimate_fps_and_rpe_keys():
    rec = _load_recorder()
    assert abs(rec.estimate_fps([0, 50, 100, 150]) - 20.0) < 1e-9
    assert rec.estimate_fps([5]) == 30.0
    assert rec.rpe_from_key(ord("7")) == 7
    assert rec.rpe_from_key(ord("0")) == 10
    assert rec.rpe_from_key(ord("s")) is None


def test_record_loop_with_fake_camera(tmp_path, monkeypatch):
    rec = _load_recorder()

    class FakeCapture:
        def __init__(self, *_):
            self.n = 0

        def isOpened(self):
            return True

        def read(self):
            time.sleep(0.05)                      # 20 fps, like a slow phone stream
            self.n += 1
            return True, np.zeros((48, 64, 3), np.uint8)

        def release(self):
            pass

    class FakeWriter:
        def __init__(self, path, fourcc, fps, size):
            FakeWriter.fps = fps

        def write(self, frame):
            pass

        def release(self):
            pass

    shown = {"n": 0}
    script = {10: ord("s"), 40: ord("e"), 42: ord("7"), 60: ord("q")}

    def wait_key(_):
        shown["n"] += 1
        return script.get(shown["n"], 255)

    fake_cv2 = types.SimpleNamespace(
        VideoCapture=FakeCapture, VideoWriter=FakeWriter, VideoWriter_fourcc=lambda *a: 0,
        imshow=lambda *a: None, waitKey=wait_key, destroyAllWindows=lambda: None,
        putText=lambda *a, **k: None, FONT_HERSHEY_SIMPLEX=0,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(rec, "session_dir", lambda *a: tmp_path)
    answers = iter(["lunges", "10", "correct"])
    monkeypatch.setattr(rec, "_ask", lambda *a, **k: next(answers))

    folder = rec.record("yassine", "s99", "fake", "none")

    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert 15 <= meta["video_fps"] <= 21           # measured, not a hard-coded 30
    assert 15 <= meta["measured_fps"] <= 21
    rows = (folder / "frames.csv").read_text(encoding="utf-8").strip().splitlines()[1:]
    ts = [int(r.split(",")[1]) for r in rows]
    assert len(rows) == meta["frames"] and ts == sorted(ts)
    sets = (folder / "sets.csv").read_text(encoding="utf-8").strip().splitlines()[1:]
    assert len(sets) == 1
    _, exercise, start, end, reps, rpe, form = sets[0].split(",")
    assert (exercise, reps, rpe, form) == ("lunges", "10", "7", "correct")
    assert 1000 <= int(end) - int(start) <= 2500   # 30 frames at 20 fps ≈ 1.5 s
