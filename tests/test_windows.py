"""Window cutting on a synthetic session, and the saved Model 1 file when it exists."""
import numpy as np
import pandas as pd
import pytest

from awc.config import MMFIT_CLASSES, PROCESSED
from awc.features.windows import make_windows


def test_window_count():
    n = 300
    classes = ["squats", "lunges", "pushups"]
    data = {
        "frame": np.arange(n),
        "exercise": [classes[i % 3] for i in range(n)],
        "session": ["syn"] * n,
        "split": ["train"] * n,
    }
    for j in range(18):
        data[f"x{j}"] = np.full(n, float(j))
        data[f"y{j}"] = np.ones(n)
    data["y1"] = np.full(n, 10.0)
    for col in ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"):
        data[col] = np.zeros(n)
    out = make_windows(pd.DataFrame(data), length=90, hop=45)
    assert len(out["y"]) == 5
    for start, label in zip((0, 45, 90, 135, 180), out["y"]):
        expected = MMFIT_CLASSES.index(classes[(start + 45) % 3])
        assert int(label) == expected


@pytest.mark.data
def test_windows_file():
    path = PROCESSED / "m1_windows.npz"
    assert path.is_file()
    z = np.load(path, allow_pickle=True)
    n = len(z["y"])
    assert z["X_pose"].shape == (n, 90, 36)
    assert z["X_imu"].shape == (n, 90, 6)
    assert z["imu_mask"].shape == (n,)
    assert z["start_frame"].shape == (n,)
    assert len(z["session"]) == n
    assert np.isfinite(z["X_pose"]).all()
    assert np.isfinite(z["X_imu"]).all()
    train = z["split"] == "train"
    present = set(int(v) for v in np.unique(z["y"][train]))
    assert present == set(range(11))
