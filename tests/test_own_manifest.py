"""Own-recordings manifest: note→exercise parser and verified table."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from awc.config import OWN_DIR, OWN_MANIFEST
from awc.data.own_manifest import (
    EXPECTED_CORE,
    build_rows,
    parse_note_exercise,
    write_manifest,
)


def test_parse_note_keywords():
    assert parse_note_exercise("PUSHU¨IHEB") == "pushups"
    assert parse_note_exercise("PUSHUP 2") == "pushups"
    assert parse_note_exercise("EX2 JUMP JACKS") == "jumping_jacks"
    assert parse_note_exercise("DUMB SHOULD PRESS") == "dumbbell_shoulder_press"
    assert parse_note_exercise("LATERAL") == "lateral_shoulder_raises"
    assert parse_note_exercise("TRICEPS") == "tricep_extensions"
    assert parse_note_exercise("TRICEP EXTENSION") == "tricep_extensions"
    assert parse_note_exercise("BICEPS 1") == "bicep_curls"
    assert parse_note_exercise("BICEP 2") == "bicep_curls"
    assert parse_note_exercise("SITUPS") == "situps"
    assert parse_note_exercise("SQUAT") == "squats"
    assert parse_note_exercise("SQUAT2") == "squats"
    assert parse_note_exercise("DUMBELL ROWS") == "dumbbell_rows"
    assert parse_note_exercise("tt") is None
    assert parse_note_exercise("") is None


@pytest.mark.data
def test_manifest_matches_verified_table():
    if not OWN_DIR.is_dir() or not any(OWN_DIR.iterdir()):
        pytest.skip("no own recordings")
    rows = build_rows(print_overrides=False)
    got = {
        (r["person"], r["session_folder"], int(r["set"])): r
        for r in rows
    }
    assert len(rows) == len(EXPECTED_CORE), (
        f"expected {len(EXPECTED_CORE)} rows, got {len(rows)}: "
        f"{sorted(got.keys())}"
    )
    for person, folder, set_n, button, exercise, reps, use, note in EXPECTED_CORE:
        key = (person, folder, set_n)
        assert key in got, f"missing {key}"
        r = got[key]
        assert r["exercise"] == exercise, f"{key}: exercise {r['exercise']!r} != {exercise!r}"
        assert r["use"] == use, f"{key}: use {r['use']!r} != {use!r}"
        got_reps = None if r["reps"] in (None, "") else int(r["reps"])
        assert got_reps == reps, f"{key}: reps {got_reps!r} != {reps!r}"
        # button / note as recorded (empty for unsaved sets)
        if button == "":
            assert (r.get("exercise_button") or "") == ""
        else:
            assert r["exercise_button"] == button
        if note:
            # note may have leading/trailing spaces in CSV
            assert note.strip() in (r.get("note") or "").strip() or (r.get("note") or "").strip() in note
        assert r["form_label"] == "correct"
        assert "hr_coverage" in r
        assert "hr_ok" in r


@pytest.mark.data
def test_write_manifest_roundtrip(tmp_path):
    if not OWN_DIR.is_dir() or not any(OWN_DIR.iterdir()):
        pytest.skip("no own recordings")
    out = tmp_path / "own_manifest.csv"
    write_manifest(path=out)
    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(EXPECTED_CORE)
    assert set(rows[0].keys()) >= {
        "person", "session_folder", "set", "exercise", "reps", "reps_typed",
        "reps_source", "use", "label_source", "hr_n", "hr_coverage", "hr_ok",
    }
    assert rows[0]["reps_source"] in ("verified", "typed_default", "unknown")


def test_expected_eval_counts():
    eval_rows = [r for r in EXPECTED_CORE if r[6] == "eval"]
    partner = [r for r in eval_rows if r[0] == "partner"]
    yassine = [r for r in eval_rows if r[0] == "yassine"]
    assert len(eval_rows) == 20
    assert len(partner) == 8
    assert len(yassine) == 12
    exercises = {r[4] for r in eval_rows}
    assert len(exercises) == 9
    assert "lunges" not in exercises


def test_manifest_prefers_verified_reps():
    from awc.data.own_manifest import apply_verified_overrides

    rows = [
        {
            "person": "yassine",
            "session_folder": "2026-11-10_SQUAT",
            "set": 1,
            "exercise": "squats",
            "reps": 10,
            "reps_typed": 10,
            "reps_source": "typed_default",
            "label_source": "button",
        },
        {
            "person": "partner",
            "session_folder": "2026-11-10_s01",
            "set": 2,
            "exercise": "pushups",
            "reps": 10,
            "reps_typed": 10,
            "reps_source": "typed_default",
            "label_source": "button",
        },
    ]
    verified = {
        ("yassine", "2026-11-10_SQUAT", 1): {
            "reps_verified": "8",
            "exercise_verified": "",
            "cant_tell": "",
        },
        ("partner", "2026-11-10_s01", 2): {
            "reps_verified": "12",
            "exercise_verified": "bicep_curls",
            "cant_tell": "",
        },
    }
    apply_verified_overrides(rows, verified)
    assert rows[0]["reps"] == 8
    assert rows[0]["reps_source"] == "verified"
    assert rows[0]["reps_typed"] == 10
    assert rows[0]["exercise"] == "squats"
    assert rows[1]["reps"] == 12
    assert rows[1]["reps_source"] == "verified"
    assert rows[1]["exercise"] == "bicep_curls"
    assert rows[1]["label_source"] == "verified"


def test_rep_metrics_use_only_verified_rows():
    """Mirror eval_own_recordings filter: main MAE uses reps_source==verified only."""
    items = [
        {"key": "a", "true": 10, "pred": 10, "reps_source": "typed_default"},
        {"key": "b", "true": 8, "pred": 9, "reps_source": "verified"},
        {"key": "c", "true": 12, "pred": 12, "reps_source": "verified"},
        {"key": "d", "true": 10, "pred": 33, "reps_source": "unknown"},
    ]
    verified = [r for r in items if r["reps_source"] == "verified"]
    typed = [r for r in items if r["reps_source"] == "typed_default"]
    assert len(verified) == 2
    assert {r["key"] for r in verified} == {"b", "c"}
    assert len(typed) == 1
    mae = sum(abs(r["true"] - r["pred"]) for r in verified) / len(verified)
    assert mae == 0.5
