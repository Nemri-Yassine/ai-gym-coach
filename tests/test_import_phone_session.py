"""Phone-app session import: a fake session JSON plus two small videos."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _importer():
    spec = importlib.util.spec_from_file_location("import_phone_session", ROOT / "scripts" / "import_phone_session.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _video(path, frames=30, fps=20):
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(frames):
        img = np.zeros((48, 64, 3), np.uint8)
        img[:, i % 64] = 255
        writer.write(img)
    writer.release()


def _session(tmp_path):
    t0 = 1_800_000_000_000
    for n in (1, 2):
        _video(tmp_path / f"awc_yassine_2026-10-27_s01_set0{n}.mp4")
    data = {
        "app": "awc-gym-recorder", "version": 1, "user": "yassine", "session_id": "s01", "date": "2026-10-27",
        "started_at_ms": t0, "ended_at_ms": t0 + 300_000, "hr_source": "bluetooth", "hr_device": "Galaxy Watch7",
        "camera": {"width": 1280, "height": 720, "frameRate": 30},
        "hr": [[t0 + 1000 * i, 90 + i % 40] for i in range(300)],
        "sets": [
            {"set": 1, "exercise": "squats", "reps": 10, "rpe": 8, "form_label": "correct",
             "start_ms": t0 + 10_000, "end_ms": t0 + 12_000, "rec_start_ms": t0 + 10_090,
             "video_file": "awc_yassine_2026-10-27_s01_set01.mp4", "frames_ms": [t0 + 10_040 + 50 * i for i in range(30)]},
            {"set": 2, "exercise": "lunges", "reps": 9, "rpe": 9, "form_label": "knee passes toe",
             "start_ms": t0 + 100_000, "end_ms": t0 + 102_000, "rec_start_ms": t0 + 100_080,
             "video_file": "awc_yassine_2026-10-27_s01_set02.mp4", "frames_ms": []},
        ],
    }
    path = tmp_path / "awc_yassine_2026-10-27_s01_session.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, t0


def test_import_writes_project_format(tmp_path):
    imp = _importer()
    src, t0 = _session(tmp_path)
    folder, meta = imp.import_session(src, out_root=tmp_path / "own")

    assert folder == tmp_path / "own" / "yassine" / "2026-10-27_s01"
    assert sorted(p.name for p in (folder / "sets").iterdir()) == ["set01.mp4", "set02.mp4"]
    hr = (folder / "hr.csv").read_text(encoding="utf-8").splitlines()
    assert hr[0] == "pc_ts_ms,bpm" and len(hr) == 301
    sets = (folder / "sets.csv").read_text(encoding="utf-8").splitlines()
    assert sets[1].split(",")[1:] == ["squats", str(t0 + 10_000), str(t0 + 12_000), "10", "8", "correct"]
    assert sets[2].split(",")[-1] == "knee passes toe"

    frames = [line.split(",") for line in (folder / "frames_set01.csv").read_text(encoding="utf-8").splitlines()[1:]]
    ts = [int(t) for _, t in frames]
    assert len(ts) == 30 and ts == sorted(ts)
    assert ts[0] == t0 + 10_040                      # anchored on the first frame the app logged
    assert abs((ts[-1] - ts[0]) - 29 * 50) <= 2       # spacing from the file: 20 fps
    second = [int(l.split(",")[1]) for l in (folder / "frames_set02.csv").read_text(encoding="utf-8").splitlines()[1:]]
    assert second[0] == t0 + 100_080                  # no logged frames: anchored on the recorder start

    assert meta["hr_samples"] == 300 and 59 <= meta["hr_samples_per_min"] <= 61
    assert meta["sets"][0]["frames_in_file"] == 30 and abs(meta["sets"][0]["fps"] - 20) < 0.5


def test_import_rejects_other_files_and_missing_videos(tmp_path):
    imp = _importer()
    bad = tmp_path / "x.json"
    bad.write_text(json.dumps({"app": "something-else"}), encoding="utf-8")
    with pytest.raises(ValueError):
        imp.import_session(bad, out_root=tmp_path / "own")
    src, _ = _session(tmp_path)
    (tmp_path / "awc_yassine_2026-10-27_s01_set02.mp4").unlink()
    with pytest.raises(FileNotFoundError):
        imp.import_session(src, out_root=tmp_path / "own2")
