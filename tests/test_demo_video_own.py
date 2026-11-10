"""Smoke checks for own-recording demo overlay."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from awc.config import OWN_DIR, OWN_MANIFEST, ROOT


def _load_demo():
    path = ROOT / "scripts" / "demo_video.py"
    spec = importlib.util.spec_from_file_location("demo_video", path)
    mod = importlib.util.module_from_spec(spec)
    # ensure sibling imports resolve if any
    sys.modules["demo_video"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_resolve_own_squat_paths():
    if not OWN_MANIFEST.is_file():
        pytest.skip("own manifest missing")
    demo = _load_demo()
    meta = demo.resolve_own("yassine/2026-11-10_SQUAT/1")
    assert meta["person"] == "yassine"
    assert meta["set"] == 1
    assert meta["video"].is_file()
    assert meta["truth_exercise"] == "squats"
    assert meta["reps_note"] in ("typed, unverified", "verified")


@pytest.mark.data
def test_render_two_seconds_own(tmp_path):
    video = OWN_DIR / "yassine" / "2026-11-10_SQUAT" / "sets" / "set01.mp4"
    if not video.is_file():
        pytest.skip("own squat video missing")
    demo = _load_demo()
    meta = demo.resolve_own("yassine/2026-11-10_SQUAT/1")
    out = tmp_path / "smoke"
    path, summary = demo.run(
        video, out, model_name="m1_transformer_pose", max_seconds=2.0,
        own_meta=meta, crf=30,
    )
    path = Path(path)
    assert path.is_file()
    assert summary["size"] == [1920, 1080]
    assert summary["frames"] > 10
    assert "timeline" in summary
    js = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert js["size"] == [1920, 1080]
