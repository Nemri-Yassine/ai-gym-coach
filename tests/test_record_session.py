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
