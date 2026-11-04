"""EC3D rep dataset: resample, labels, leave-one-person-out folds."""
import numpy as np
import pytest

from awc.features.form_dataset import (
    EXERCISES,
    MISTAKE_CODES,
    N_MISTAKES,
    REP_FRAMES,
    TEST_YAW,
    TRAIN_YAWS,
    FormDataset,
    exercise_counts,
    fold_indices,
    is_correct,
    loso_folds,
    mistake_index,
    resample,
)


def test_resample_length():
    rng = np.random.default_rng(0)
    for n in (1, 2, 7, 24, 64, 100, 208):
        seq = rng.normal(size=(n, 18, 2))
        out = resample(seq, REP_FRAMES)
        assert out.shape == (REP_FRAMES, 18, 2)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out[0], seq[0], atol=1e-5)
        np.testing.assert_allclose(out[-1], seq[-1], atol=1e-5)


def test_no_subject_leak():
    subjects = np.array(["Hugues", "Isinsu", "Sena", "Vidit", "Hugues"])
    folds = loso_folds(subjects)
    assert len(folds) == 4
    held_out = []
    for fold in folds:
        train, test = fold_indices(subjects, fold["test"])
        train_people = set(subjects[train].tolist())
        test_people = set(subjects[test].tolist())
        assert train_people.isdisjoint(test_people)
        assert test_people == {fold["test"]}
        assert fold["test"] not in fold["train"]
        assert set(fold["train"]) | {fold["test"]} == set(subjects.tolist())
        held_out.append(fold["test"])
    assert sorted(held_out) == sorted(set(subjects.tolist()))


def test_label_spaces():
    assert [mistake_index("squat", code) for code in MISTAKE_CODES["squat"]] == [0, 1, 2, 3, 4]
    assert [mistake_index("lunge", code) for code in MISTAKE_CODES["lunge"]] == [0, 1, 2]
    assert [mistake_index("plank", code) for code in MISTAKE_CODES["plank"]] == [0, 1, 2]
    for exercise, n_class in N_MISTAKES.items():
        labels = [mistake_index(exercise, code) for code in MISTAKE_CODES[exercise]]
        assert min(labels) == 0
        assert max(labels) == n_class - 1
        assert n_class == {"squat": 5, "lunge": 3, "plank": 3}[exercise]
    assert is_correct("1") == 1
    assert is_correct("4") == 0
    assert is_correct("10") == 0
    assert EXERCISES == ("squat", "lunge", "plank")


def _tpose(n=12):
    pose = np.zeros((n, 3, 25), dtype=np.float64)
    pose[:, 0, 19] = pose[:, 0, 22] = 0.3
    pose[:, 0, 21] = pose[:, 0, 24] = -0.1
    pose[:, 1, 2] = 0.4
    pose[:, 1, 5] = -0.4
    pose[:, 2, 1] = 1.0
    pose[:, 0, 9] = 0.2
    pose[:, 0, 12] = -0.2
    return pose


def _synthetic_pack():
    pose = _tpose()
    n = 4
    poses = np.zeros((n, pose.shape[0], 3, 25), dtype=np.float32)
    poses[:] = pose
    return {
        "poses": poses,
        "length": np.full(n, pose.shape[0], dtype=np.int32),
        "exercise": np.array(["squat", "lunge", "plank", "squat"]),
        "subject": np.array(["Hugues", "Isinsu", "Sena", "Vidit"]),
        "mistake": np.array([0, 1, 2, 4], dtype=np.int64),
        "is_correct": np.array([1, 0, 0, 0], dtype=np.int64),
        "rep": np.arange(n, dtype=np.int64),
        "mistake_code": np.array(["1", "4", "8", "5"]),
    }


def test_views_train_and_test():
    pack = _synthetic_pack()
    train = FormDataset(pack, [0, 1], view="train", seed=0)
    test = FormDataset(pack, [2], view="test", seed=0)
    sweep = FormDataset(pack, [2], view="sweep", seed=0)
    assert len(train) == 2 * len(TRAIN_YAWS) * 2
    assert len(test) == 1
    assert len(sweep) == len(TRAIN_YAWS)
    item = test[0]
    assert item["pose"].shape == (REP_FRAMES, 18, 2)
    assert item["yaw"] == TEST_YAW
    assert item["mirrored"] is False
    assert item["subject"] == "Sena"
    np.testing.assert_array_equal(test[0]["pose"], test[0]["pose"])
    yaws = {train[i]["yaw"] for i in range(len(train))}
    mirrors = {train[i]["mirrored"] for i in range(len(train))}
    assert yaws == set(TRAIN_YAWS)
    assert mirrors == {False, True}
    assert not np.allclose(train[0]["pose"], train[0]["pose"])


@pytest.mark.data
def test_rep_counts():
    from awc.features.form_dataset import build_m2_reps, load_m2_reps

    pack = build_m2_reps()
    loaded = load_m2_reps()
    assert len(pack["exercise"]) == 362
    counts = exercise_counts(pack["exercise"])
    assert counts == {"squat": 132, "lunge": 127, "plank": 103}
    assert exercise_counts(loaded["exercise"]) == counts
    assert "10" not in set(loaded["mistake_code"].astype(str))
    for exercise, n_class in N_MISTAKES.items():
        labels = loaded["mistake"][loaded["exercise"].astype(str) == exercise]
        assert labels.min() >= 0
        assert labels.max() <= n_class - 1
    folds = loso_folds(loaded["subject"])
    assert len(folds) == 4
    for fold in folds:
        train, test = fold_indices(loaded["subject"], fold["test"])
        assert set(loaded["subject"][train].astype(str)).isdisjoint(
            set(loaded["subject"][test].astype(str))
        )
    squat = int(np.flatnonzero(loaded["exercise"].astype(str) == "squat")[0])
    sample = FormDataset(loaded, [squat], view="test")[0]
    assert sample["pose"].shape == (REP_FRAMES, 18, 2)
    assert np.isfinite(sample["pose"]).all()
    # y is up: on a squat the neck stays above the ankles at 45°.
    assert float(sample["pose"][:, 1, 1].mean()) > float(sample["pose"][:, 10, 1].mean())
