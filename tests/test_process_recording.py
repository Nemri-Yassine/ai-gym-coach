"""process_recording: synthetic video + portrait invariance + real session load."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from awc.config import OWN_DIR
from awc.data import recordings
from awc.features.pose_video import landmarks_to_pixels
from awc.features.skeleton import normalize


def test_portrait_and_padded_landscape_normalize_match():
    """Same body with true portrait size vs letterboxed square → same after normalize."""
    rng = np.random.default_rng(42)
    # 33 MediaPipe-style normalized landmarks (standing person-ish)
    lm = np.zeros((33, 2), dtype=np.float64)
    lm[:, 0] = 0.5 + rng.normal(0, 0.05, 33)
    lm[:, 1] = np.linspace(0.1, 0.9, 33) + rng.normal(0, 0.01, 33)
    # shoulders / hips for a usable torso
    lm[11] = [0.40, 0.35]
    lm[12] = [0.60, 0.35]
    lm[23] = [0.42, 0.55]
    lm[24] = [0.58, 0.55]

    portrait = landmarks_to_pixels(lm, 720, 1280)
    pad = (1280 - 720) / 2.0
    lm_land = lm.copy()
    lm_land[:, 0] = (pad + lm[:, 0] * 720.0) / 1280.0
    landscape = landmarks_to_pixels(lm_land, 1280, 1280)

    a = normalize(portrait[None])[0]
    b = normalize(landscape[None])[0]
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_join_hr_nearest_synthetic():
    from awc.features.own_sync import join_hr_nearest

    pc = np.asarray([1000.0, 2000.0, 5000.0])
    hr_rows = [(1005, 90.0), (2100, 100.0)]
    hr, mask = join_hr_nearest(pc, hr_rows, max_ms=2000)
    assert mask[0] == 1 and hr[0] == 90.0
    assert mask[1] == 1 and hr[1] == 100.0
    assert mask[2] == 0 and not np.isfinite(hr[2])


def test_process_synthetic_video(tmp_path, monkeypatch):
    """Tiny synthetic MP4 + frames.csv → parquet with increasing timestamps and HR mask."""
    import importlib.util
    import sys

    import cv2

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "process_recording", root / "scripts" / "process_recording.py"
    )
    pr = importlib.util.module_from_spec(spec)
    sys.modules["process_recording"] = pr
    spec.loader.exec_module(pr)

    person, session, set_n = "yassine", "2099-01-01_SYN", 1
    session_dir = tmp_path / person / session
    (session_dir / "sets").mkdir(parents=True)
    video = session_dir / "sets" / "set01.mp4"
    n = 30
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 96))
    for i in range(n):
        frame = np.full((96, 64, 3), 40, dtype=np.uint8)
        cv2.circle(frame, (32, 20 + (i % 10)), 8, (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    assert video.stat().st_size > 0

    times = [1_000_000 + i * 33 for i in range(n)]
    pd.DataFrame({"frame": range(n), "pc_ts_ms": times}).to_csv(
        session_dir / "frames_set01.csv", index=False
    )
    pd.DataFrame({
        "pc_ts_ms": [1_000_000 + 100, 1_000_000 + 500],
        "bpm": [100.0, 110.0],
    }).to_csv(session_dir / "hr.csv", index=False)

    interim = tmp_path / "interim" / "own"
    monkeypatch.setattr(pr, "OWN_DIR", tmp_path)
    monkeypatch.setattr(pr, "OWN_INTERIM", interim)
    monkeypatch.setattr(pr, "interim_path",
                        lambda p, s, n: interim / p / f"{s}_set{n:02d}.parquet")
    monkeypatch.setattr(pr, "qa_path",
                        lambda p, s, n: interim / p / f"{s}_set{n:02d}_qa.json")

    pose = np.zeros((n, 18, 2), dtype=np.float32)
    pose[:, 8, 1] = np.linspace(0, 1, n)
    pose[:, 11, 1] = np.linspace(0, 1, n)
    pose[:, 1, 1] = np.linspace(0.5, 1.5, n)

    def fake_extract(video_path, frame_times_ms=None, return_qa=False, **kw):
        qa = {"pct_detected": 1.0, "mean_visibility": 0.9, "n_gaps_filled": 0,
              "n_src": n, "n_out": n}
        if return_qa:
            return pose, 30.0, n, qa
        return pose, 30.0, n

    import awc.features.pose_video as pv
    monkeypatch.setattr(pv, "extract_pose", fake_extract)

    row = {
        "person": person,
        "session_folder": session,
        "set": set_n,
        "video": "sets/set01.mp4",
        "exercise": "squats",
    }
    result = pr.process_one(row, force=True)
    assert result["status"] == "ok"
    out = interim / person / f"{session}_set{set_n:02d}.parquet"
    assert out.is_file()
    df = pd.read_parquet(out)
    assert len(df) == n
    assert df["pc_ts_ms"].is_monotonic_increasing
    assert "hr" in df.columns and "hr_mask" in df.columns
    assert int(df["hr_mask"].sum()) >= 1
    assert list(df["x0"].values)


@pytest.mark.data
def test_recordings_load_real_session():
    session = OWN_DIR / "yassine" / "2026-11-10_SQUAT"
    if not session.is_dir():
        pytest.skip("no own recordings")
    data = recordings.load(session)
    assert data["meta"].get("user") == "yassine"
    assert len(data["sets"]) >= 2
    assert 1 in data["frames"] and len(data["frames"][1]) > 10
    assert data["sets"][0]["video"]
    assert Path(data["sets"][0]["video"]).is_file()


@pytest.mark.data
def test_recordings_load_orphan_session():
    session = OWN_DIR / "yassine" / "2026-11-10_PUHSUP2"
    if not session.is_dir():
        pytest.skip("no own recordings")
    data = recordings.load(session)
    assert len(data["sets"]) >= 1
    assert data["sets"][0]["set"] == 1
    assert data["sets"][0]["video"]
