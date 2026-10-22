"""EC3D loader (Exercise Correction in 3D, Zhao et al., ACCV 2022).

data_3D.pickle = {'labels': (29789, 5) str, 'poses': (29789, 3, 25) float}
- labels columns: exercise, subject, mistake code, rep index, frame
- poses: x, y, z (z up) for 25 OpenPose BODY_25 joints, already normalized by the authors
  (mid-hip at the origin, trunk upright, equal bone lengths)
One (exercise, subject, mistake, rep) group = one rep with one label (371 reps).
"""
import collections
import pickle
import pickletools

from awc.config import EC3D_DIR

# mistake codes, from the authors' evaluation.py
MISTAKES = {"1": "Correct", "2": "Feet too wide", "3": "Knees inward", "4": "Not low enough", "5": "Front bent",
            "6": "Knee passes toe", "7": "Arched back", "8": "Hunched back", "9": "Asymmetric", "10": "Unknown"}

JOINTS = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist", "mid_hip", "r_hip",
          "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear", "l_big_toe",
          "l_small_toe", "l_heel", "r_big_toe", "r_small_toe", "r_heel"]

_SAFE = {"numpy dtype", "numpy ndarray", "numpy.core.multiarray _reconstruct"}


def _check_pickle(path):
    """Pickles can run code when loaded: refuse anything that references more than plain numpy arrays."""
    with open(path, "rb") as f:
        refs = {str(arg) for op, arg, _ in pickletools.genops(f) if op.name == "GLOBAL"}
    if not refs <= _SAFE:
        raise ValueError(f"{path.name} references unexpected objects: {refs - _SAFE}")


def load(name="data_3D.pickle"):
    path = EC3D_DIR / name
    _check_pickle(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def reps(data=None):
    """Group frames into reps: {(exercise, subject, mistake, rep): [row indices]}."""
    data = data or load()
    out = collections.OrderedDict()
    for i, key in enumerate(map(tuple, data["labels"][:, :4])):
        out.setdefault(key, []).append(i)
    return out
