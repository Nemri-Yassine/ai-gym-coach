"""MM-Fit loader.

Layout: data/raw/mm-fit/wXX/wXX_<stream>.npy plus wXX_labels.csv.
- IMU / heart-rate files: rows of [frame, timestamp_ms, values...]
- pose_2d: shape (2, frames, 19) -> x/y, then [frame, 18 joints] (OpenPose COCO-18 order)
- pose_3d: shape (3, frames, 18) -> x/y/z, then [frame, 17 joints]
- labels.csv: one row per set: start_frame, end_frame, reps, exercise
The shared `frame` column (30 fps video clock) is the sync key between all streams.
Heart rate is missing in sessions w05, w10 and w18.
"""
import csv
import numpy as np

from awc.config import MMFIT_DIR

JOINTS_2D = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
             "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]

# streams we keep for our setup (phone films, one watch); see the Capture slide
USED_STREAMS = ["pose_2d", "sw_l_acc", "sw_l_gyr", "sw_l_hr"]


def sessions():
    """Session ids, e.g. ['w00', 'w01', ...]."""
    return sorted(p.name for p in MMFIT_DIR.glob("w*") if p.is_dir())


def load_stream(session, stream, mmap=True):
    """Load one stream of a session, e.g. load_stream('w00', 'sw_l_acc'). Returns None if the file is missing."""
    path = MMFIT_DIR / session / f"{session}_{stream}.npy"
    if not path.exists():
        return None
    return np.load(path, mmap_mode="r" if mmap else None)


def load_labels(session):
    """List of (start_frame, end_frame, reps, exercise) for every set of a session."""
    with open(MMFIT_DIR / session / f"{session}_labels.csv") as f:
        return [(int(a), int(b), int(r), ex) for a, b, r, ex in csv.reader(f)]
