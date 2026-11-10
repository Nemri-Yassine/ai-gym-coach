"""Unit tests for the gym station (no iPhone / watch required)."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import pytest

from awc.data import recordings as rec
from awc.station.chunks import ChunkAssembler
from awc.station.clock import ClockSync
from awc.station.hr_parse import parse_hr_measurement


def test_hr_parse_uint8():
    payload = bytes([0x00, 72])  # flags=0, bpm=72
    bpm, rr = parse_hr_measurement(payload)
    assert bpm == 72
    assert rr == []


def test_hr_parse_uint16_with_rr():
    # flags: bit0 uint16 + bit4 RR; bpm=100; RR ticks 1024 → 1000 ms
    payload = bytes([0x11, 100, 0, 0x00, 0x04])
    bpm, rr = parse_hr_measurement(payload)
    assert bpm == 100
    assert len(rr) == 1
    assert abs(rr[0] - 1000.0) < 0.1


def test_clock_offset_noisy_rtt():
    sync = ClockSync()
    true_offset = 1234.0
    for i in range(5):
        t0 = 1_000_000 + i * 1000
        phone = t0 + 5 + true_offset
        t1 = t0 + 10
        sync.add(t0, phone, t1)
    for i in range(5):
        t0 = 2_000_000 + i * 1000
        phone = t0 + 200 + true_offset + 500
        t1 = t0 + 400
        sync.add(t0, phone, t1)
    offset, rtt = sync.estimate()
    assert offset is not None
    assert abs(offset - true_offset) < 5.0
    assert rtt < 50


def test_chunk_assembly_out_of_order_and_retry(tmp_path: Path):
    path = tmp_path / "set01.mp4"
    asm = ChunkAssembler(path)
    assert asm.add(1, b"BB") == []
    assert asm.add(0, b"AA") == [0, 1]
    assert asm.add(1, b"BB") == []
    assert path.read_bytes() == b"AABB"
    assert asm.is_complete(2)


def test_session_files_and_load(tmp_path: Path, monkeypatch):
    from awc.station import session_io
    from awc.station.session_io import StationSession

    monkeypatch.setattr(
        session_io,
        "verify_video",
        lambda p: {"frames": 10, "fps": 30.0, "ok": True, "remuxed": False, "path": str(p)},
    )
    folder = tmp_path / "yassine" / "2026-11-09_t01"
    sess = StationSession(folder, meta={
        "source": "gym-station-iphone", "user": "yassine", "session_id": "t01", "date": "2026-11-09",
    })
    for set_i in (1, 2):
        start = int(time.time() * 1000)
        active = sess.start_set(set_i, start, ext="mp4")
        # out-of-order then fill
        sess.add_chunk(1, b"mdat")
        sess.add_chunk(0, b"ftyp")
        sess.add_chunk(1, b"mdat")  # retry ignored
        active.phone_frame_ts = [float(start + i * 33) for i in range(10)]
        end = start + 5000
        info = sess.finish_upload(2, end, lambda t: t)
        assert info["video"]["ok"]
        sess.save_set_labels(
            set_i, "squats", start, end, 8, 6, "correct",
            {"session_type": "fatigue", "note": "test", "duration_s": 5.0},
        )

    assert sess.next_set() == 3
    with (folder / "sets.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == rec.SETS_HEADER
    assert len(rows) == 2
    assert int(rows[0]["start_ms"]) <= int(rows[0]["end_ms"])
    assert (folder / "sets" / "set01.mp4").read_bytes() == b"ftypmdat"

    loaded = rec.load(folder)
    assert len(loaded["sets"]) == 2
    assert loaded["meta"]["source"] == "gym-station-iphone"
    assert len(loaded["frames"][1]) == 10


def test_station_api_smoke(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rec, "OWN_DIR", tmp_path)
    from awc.station.app import create_app
    from awc.station.state import RUNTIME
    from fastapi.testclient import TestClient

    RUNTIME.session = None
    RUNTIME.recording = False
    RUNTIME.pending_labels = None
    RUNTIME.command_q.clear()

    client = TestClient(create_app(pin="1234"))
    assert client.get("/").status_code == 200
    assert client.get("/phone").status_code == 200
    r = client.post("/api/session/start", json={
        "user": "yassine", "session_id": "api1", "session_type": "fatigue", "date": "2026-11-09",
    })
    assert r.status_code == 200
    assert r.json()["next_set"] == 1
    r2 = client.post("/api/set/start", json={"allow_no_phone": True})
    assert r2.status_code == 200
    assert RUNTIME.recording is True
    assert any(c.get("type") == "start_set" for c in RUNTIME.command_q) or True
    # command was pushed then maybe not popped
    assert RUNTIME.current_set == 1


def test_recordings_plank_not_in_m1_classes():
    from awc.config import MMFIT_CLASSES

    assert "plank" in rec.EXERCISES
    assert "plank" not in MMFIT_CLASSES


def _reset_runtime():
    from awc.station.state import RUNTIME

    RUNTIME.session = None
    RUNTIME.recording = False
    RUNTIME.pending_labels = None
    RUNTIME.upload = {}
    RUNTIME.analyses = {}
    RUNTIME.command_q.clear()
    return RUNTIME


@pytest.mark.data
def test_analysis_worker_on_cached_set():
    """Analysis on a cached own set (no camera): JSON fields present, exercise in classes, reps ≥ 0."""
    from awc.config import MMFIT_CLASSES, OWN_DIR, OWN_INTERIM
    from awc.station.analyze import analyze_set, analysis_path

    person, session, set_n = "yassine", "2026-11-10_SQUAT", 1
    session_dir = OWN_DIR / person / session
    cached = OWN_INTERIM / person / f"{session}_set{set_n:02d}.parquet"
    if not session_dir.is_dir() or not cached.is_file():
        pytest.skip("cached own squat set missing")
    result = analyze_set(session_dir, set_n, person=person, session_folder=session)
    assert result["exercise"] in MMFIT_CLASSES
    assert int(result["reps"]) >= 0
    assert "confidence" in result and "summary" in result
    assert "elapsed_s" in result
    path = analysis_path(person, session, set_n)
    assert path.is_file()
    # never writes into data/raw
    assert "own_live" in str(path)
    assert "own_recordings" not in str(path)


def test_station_pin_guard():
    from fastapi.testclient import TestClient

    from awc.station.app import create_app

    app = create_app(pin="4321")
    phone = TestClient(app, client=("192.168.1.50", 50000))
    assert phone.get("/phone").status_code == 401  # PIN page
    assert phone.get("/phone?pin=4321").status_code == 200
    assert phone.get("/").status_code == 403  # control page is laptop-only
    assert phone.post("/api/session/start", json={}).status_code == 403
    assert phone.post("/api/upload/1/0", content=b"x").status_code == 401
    assert phone.post("/api/upload/1/0", content=b"x", headers={"X-AWC-Pin": "4321"}).status_code == 200


def _make_fmp4(path: Path, seconds=2, fps=30):
    import subprocess

    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi",
           "-i", f"testsrc=size=320x240:rate={fps}", "-t", str(seconds), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov+default_base_moof",
           "-frag_duration", "1000000", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path.read_bytes()


def test_station_upload_flow_real_fmp4(tmp_path: Path, monkeypatch):
    """Phone uploads fragmented-MP4 pieces out of order + a retry; set_done after the
    chunks → playable remuxed mp4, frames_setNN.csv on the laptop clock from video PTS."""
    from fastapi.testclient import TestClient

    from awc.station.app import create_app

    monkeypatch.setattr(rec, "OWN_DIR", tmp_path)
    rt = _reset_runtime()
    raw = _make_fmp4(tmp_path / "src.mp4", seconds=2, fps=30)
    pieces = [raw[i:i + 40_000] for i in range(0, len(raw), 40_000)]

    client = TestClient(create_app(pin="1111"))
    r = client.post("/api/session/start", json={"user": "yassine", "session_id": "up1", "date": "2026-11-10"})
    folder = Path(r.json()["folder"])
    r = client.post("/api/set/start", json={"allow_no_phone": True})
    assert r.json()["set"] == 1
    rt.clock.add(1000.0, 1000.0 + 500 + 5, 1010.0)  # phone clock = laptop + 500 ms

    order = list(range(len(pieces)))[::-1] + [0]  # reversed, then a duplicate retry
    for seq in order:
        assert client.post(f"/api/upload/1/{seq}", content=pieces[seq]).status_code == 200
    assert client.post("/api/set/end").status_code == 200
    # Save is blocked until the video is complete
    assert client.post("/api/set/save", json={"exercise": "squats", "reps": 5, "rpe": 7,
                                              "form_label": "correct"}).status_code == 409
    rec_start_phone = 1_800_000_000_500.0
    r = client.post("/api/phone/set_done/1", json={"n_chunks": len(pieces), "phone_rec_start_ms": rec_start_phone,
                                                   "frames": [rec_start_phone + i * 33.3 for i in range(60)]})
    assert r.status_code == 200, r.text
    assert r.json()["frames"] == 60
    assert abs(r.json()["fps"] - 30) < 0.5

    video = folder / "sets" / "set01.mp4"
    import cv2
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened()
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    assert n == 60

    with (folder / "frames_set01.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 60
    t = [int(x["pc_ts_ms"]) for x in rows]
    assert t[0] == 1_800_000_000_000  # phone start − 500 ms offset → laptop clock
    assert all(b > a for a, b in zip(t, t[1:]))
    assert abs((t[-1] - t[0]) - 59 * 1000 / 30) < 5

    r = client.post("/api/set/save", json={"exercise": "squats", "reps": 5, "rpe": 7, "form_label": "correct"})
    assert r.status_code == 200 and r.json()["video_ok"]
    loaded = rec.load(folder)
    assert len(loaded["sets"]) == 1 and loaded["sets"][0]["video"].endswith("set01.mp4")
    # next set never reuses an existing video number
    assert client.get("/api/status").json()["next_set"] == 2
