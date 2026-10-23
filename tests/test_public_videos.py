"""Video → skeleton helpers, RepCount label mapping, and the cached RepCount skeletons when present."""
import numpy as np
import pytest

from awc.config import INTERIM, REPCOUNT_DIR
from awc.data.repcount import exercise_name
from awc.features.pose_video import fill_gaps, landmarks_to_pixels, resample


def test_exercise_name_mapping():
    assert exercise_name("squant") == "squats"
    assert exercise_name("jumpjacks") == "jumping_jacks"
    assert exercise_name("pullups") == "pull_up"
    assert exercise_name("pommelhorse") is None


def test_landmarks_to_pixels_y_up():
    lm = np.full((33, 2), 0.5)
    lm[0] = [0.5, 0.1]    # nose near the top of the image
    lm[27] = [0.4, 0.9]   # left ankle near the bottom
    lm[28] = [0.6, 0.9]
    out = landmarks_to_pixels(lm, width=640, height=360)
    assert out.shape == (18, 2)
    assert out[0, 1] > out[10, 1]
    assert out[0, 0] == pytest.approx(320)


def test_fill_gaps_short_and_long():
    seq = np.zeros((20, 18, 2))
    seq[:, :, 0] = np.arange(20)[:, None]
    seq[5:7] = np.nan     # 2 frames: filled
    seq[10:16] = np.nan   # 6 frames: left missing
    out = fill_gaps(seq, max_gap=3)
    assert np.allclose(out[5, :, 0], 5)
    assert np.isnan(out[12]).all()


def test_resample_25_to_30():
    t = np.arange(50) / 25
    seq = np.zeros((50, 18, 2))
    seq[:, :, 1] = t[:, None]
    out = resample(seq, 25, 30)
    assert len(out) == int(np.floor((49 / 25) * 30)) + 1
    assert np.allclose(out[:, 0, 1], np.arange(len(out)) / 30)


@pytest.mark.data
def test_repcount_cache():
    if not (REPCOUNT_DIR / "test.lance").exists():
        pytest.skip("RepCount-A not downloaded")
    files = sorted((INTERIM / "repcount" / "pose").glob("*.npz"))
    if not files:
        pytest.skip("run scripts/eval_public_videos.py --extract first")
    z = np.load(files[0])
    assert z["pose"].ndim == 3 and z["pose"].shape[1:] == (18, 2)
    assert float(z["fps"]) > 1
