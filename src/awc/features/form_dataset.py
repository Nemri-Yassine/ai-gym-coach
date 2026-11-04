"""EC3D reps as fixed-length 2D skeletons for Model 2.

Each rep is stored once, as its 3D pose (OpenPose BODY_25). The PyTorch
dataset applies the camera angle with project_ec3d, then normalize, then a
resample to 64 frames. Mirroring and the train-only augmentation happen
there too, so the npz does not store a copy per angle.
"""
import numpy as np
from torch.utils.data import Dataset

from awc.config import M2_REPS
from awc.data.ec3d import load as load_ec3d
from awc.data.ec3d import reps as ec3d_reps
from awc.features.skeleton import mirror, normalize, project_ec3d

REP_FRAMES = 64
TRAIN_YAWS = (0, 15, 30, 45, 60, 75, 90)
TEST_YAW = 45
UNKNOWN_CODE = "10"

# Names used everywhere after the loader. EC3D spells them SQUAT / Lunges / Plank.
EXERCISES = ("squat", "lunge", "plank")
EC3D_EXERCISE = {"SQUAT": "squat", "Lunges": "lunge", "Plank": "plank"}

# Mistake index inside the exercise. Correct is always 0. Code 10 is dropped.
MISTAKE_CODES = {
    "squat": ("1", "2", "3", "4", "5"),
    "lunge": ("1", "4", "6"),
    "plank": ("1", "7", "8"),
}
N_MISTAKES = {name: len(codes) for name, codes in MISTAKE_CODES.items()}
_MISTAKE_INDEX = {
    name: {code: i for i, code in enumerate(codes)} for name, codes in MISTAKE_CODES.items()
}

_AUG_SCALE = (0.9, 1.1)
_AUG_WARP = (0.8, 1.2)
_AUG_NOISE = 0.01


def resample(seq, length=REP_FRAMES) -> np.ndarray:
    """Linear interpolation over time. Shape (T, ...) → (length, ...); endpoints stay."""
    seq = np.asarray(seq, dtype=np.float64)
    n = int(seq.shape[0])
    if n == 0:
        raise ValueError("cannot resample an empty sequence")
    flat = seq.reshape(n, -1)
    if n == 1:
        out = np.repeat(flat, length, axis=0)
    else:
        pos = np.linspace(0.0, n - 1, length)
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, n - 1)
        weight = (pos - i0)[:, None]
        out = flat[i0] * (1.0 - weight) + flat[i1] * weight
    return out.reshape((length,) + seq.shape[1:]).astype(np.float32)


def time_warp(seq, factor) -> np.ndarray:
    """Stretch or compress time by `factor`, then resample back to the same length."""
    n = int(np.asarray(seq).shape[0])
    stretched = max(2, int(round(n * float(factor))))
    return resample(resample(seq, stretched), n)


def mistake_index(exercise, code) -> int:
    """Index of a mistake code inside its exercise (squat 0..4, lunge 0..2, plank 0..2)."""
    try:
        return _MISTAKE_INDEX[exercise][str(code)]
    except KeyError:
        raise KeyError(f"no mistake index for {exercise} code {code}") from None


def is_correct(code) -> int:
    return int(str(code) == "1")


def loso_folds(subjects):
    """Leave-one-person-out. One fold per subject; that person is absent from train."""
    names = tuple(sorted({str(s) for s in subjects}))
    return [
        {"fold": i, "test": held, "train": tuple(s for s in names if s != held)}
        for i, held in enumerate(names)
    ]


def fold_indices(subjects, test_subject):
    """Row indices for one fold. Train and test share no person."""
    subjects = np.asarray(subjects).astype(str)
    test = np.flatnonzero(subjects == str(test_subject))
    train = np.flatnonzero(subjects != str(test_subject))
    return train, test


def _pack_rows(rows):
    lengths = np.array([r["pose"].shape[0] for r in rows], dtype=np.int32)
    tmax = int(lengths.max()) if len(rows) else 0
    poses = np.zeros((len(rows), tmax, 3, 25), dtype=np.float32)
    for i, row in enumerate(rows):
        poses[i, : lengths[i]] = row["pose"]
    return {
        "poses": poses,
        "length": lengths,
        "exercise": np.array([r["exercise"] for r in rows]),
        "subject": np.array([r["subject"] for r in rows]),
        "mistake": np.array([r["mistake"] for r in rows], dtype=np.int64),
        "is_correct": np.array([r["is_correct"] for r in rows], dtype=np.int64),
        "rep": np.array([r["rep"] for r in rows], dtype=np.int64),
        "mistake_code": np.array([r["mistake_code"] for r in rows]),
    }


def iter_kept_reps(data=None):
    """One dict per rep. Code 10 (unknown) is dropped."""
    data = data or load_ec3d()
    poses = data["poses"]
    for (exercise, subject, code, rep), rows in ec3d_reps(data).items():
        code = str(code)
        if code == UNKNOWN_CODE:
            continue
        name = EC3D_EXERCISE[str(exercise)]
        yield {
            "exercise": name,
            "subject": str(subject),
            "mistake": mistake_index(name, code),
            "mistake_code": code,
            "is_correct": is_correct(code),
            "rep": int(rep),
            "pose": poses[rows],
        }


def exercise_counts(exercise) -> dict:
    exercise = np.asarray(exercise).astype(str)
    return {name: int((exercise == name).sum()) for name in EXERCISES}


def build_m2_reps(data=None, path=None):
    """Build the rep table and write `data/processed/m2_reps.npz`. Returns the arrays."""
    pack = _pack_rows(list(iter_kept_reps(data)))
    path = M2_REPS if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pack)
    counts = exercise_counts(pack["exercise"])
    lengths = pack["length"]
    print(
        f"m2 reps {len(lengths)}: "
        + ", ".join(f"{name} {counts[name]}" for name in EXERCISES)
        + f"; subjects {len(set(pack['subject'].astype(str)))}; "
        + f"frames min/median/max {int(lengths.min())}/"
        + f"{int(np.median(lengths))}/{int(lengths.max())}"
    )
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    return pack


def load_m2_reps(path=None):
    path = M2_REPS if path is None else path
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def rep_pose(pack, index) -> np.ndarray:
    """3D sequence for one rep, without the zero padding: (T, 3, 25)."""
    n = int(pack["length"][index])
    return np.asarray(pack["poses"][index, :n])


def skeleton_from_rep(pose3d, yaw_deg, mirrored=False) -> np.ndarray:
    """project_ec3d → normalize → 64 frames, then optional mirror. (64, 18, 2)."""
    seq = resample(normalize(project_ec3d(pose3d, yaw_deg)), REP_FRAMES)
    if mirrored:
        seq = mirror(seq)
    return seq


def augment_skeleton(seq, rng) -> np.ndarray:
    """Train-only: scale U(0.9, 1.1), time-warp U(0.8, 1.2), Gaussian noise σ=0.01."""
    scale = float(rng.uniform(*_AUG_SCALE))
    seq = np.asarray(seq, dtype=np.float64) * scale
    factor = float(rng.uniform(*_AUG_WARP))
    seq = time_warp(seq, factor)
    seq = seq + rng.normal(0.0, _AUG_NOISE, size=seq.shape)
    return seq.astype(np.float32)


class FormDataset(Dataset):
    """One item is one rep at one camera angle.

    view='train': yaw in {0,15,30,45,60,75,90} × {normal, mirrored}, with augmentation.
    view='test': 45° only, no mirror, no augmentation.
    view='sweep': the seven yaws, no mirror, no augmentation (robustness check).
    """

    def __init__(self, pack, indices, view="train", seed=42):
        if view not in ("train", "test", "sweep"):
            raise ValueError(f"unknown view {view}")
        self.pack = pack
        self.indices = np.asarray(indices, dtype=np.int64)
        self.view = view
        self.augment = view == "train"
        if view == "train":
            self.views = [(yaw, mirrored) for yaw in TRAIN_YAWS for mirrored in (False, True)]
        elif view == "test":
            self.views = [(TEST_YAW, False)]
        else:
            self.views = [(yaw, False) for yaw in TRAIN_YAWS]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return int(len(self.indices) * len(self.views))

    def __getitem__(self, index):
        n_views = len(self.views)
        row = int(self.indices[index // n_views])
        yaw, mirrored = self.views[index % n_views]
        pose = skeleton_from_rep(rep_pose(self.pack, row), yaw, mirrored)
        if self.augment:
            pose = augment_skeleton(pose, self.rng)
        return {
            "pose": pose,
            "exercise": str(self.pack["exercise"][row]),
            "mistake": int(self.pack["mistake"][row]),
            "is_correct": int(self.pack["is_correct"][row]),
            "subject": str(self.pack["subject"][row]),
            "yaw": int(yaw),
            "mirrored": bool(mirrored),
        }


def main():
    build_m2_reps()


if __name__ == "__main__":
    main()
