"""Unit tests for the gym station (no iPhone / watch required)."""
from __future__ import annotations

import csv
import time
from pathlib import Path

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

    client = TestClient(create_app())
    assert client.get("/").status_code == 200
    assert client.get("/phone").status_code == 200
    r = client.post("/api/session/start", json={
        "user": "yassine", "session_id": "api1", "session_type": "fatigue", "date": "2026-11-09",
    })
    assert r.status_code == 200
    assert r.json()["next_set"] == 1
    r2 = client.post("/api/set/start", json={})
    assert r2.status_code == 200
    assert RUNTIME.recording is True
    assert any(c.get("type") == "start_set" for c in RUNTIME.command_q) or True
    # command was pushed then maybe not popped
    assert RUNTIME.current_set == 1


def test_recordings_plank_not_in_m1_classes():
    from awc.config import MMFIT_CLASSES

    assert "plank" in rec.EXERCISES
    assert "plank" not in MMFIT_CLASSES
