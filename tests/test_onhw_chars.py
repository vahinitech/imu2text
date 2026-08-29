"""Tests for the OnHW-chars dataset loader (onhw_chars.py).

Covers both the .pkl format (left-handed, no splits) and the .npy format
(right-handed, 30 official splits). The .pkl tests use the small (3.5 MB)
OnHW-chars_L dataset that CI can download on the fly; the .npy tests
synthetically generate a tiny mock of the .npy folder structure so they
don't need the 896 MB download.
"""

import os
import pickle
import shutil
import tempfile

import numpy as np
import pytest

import onhw_chars as C


# --------------------------------------------------------------------------- #
# Synthetic .pkl dataset (mirrors the OnHW-chars_L layout)
# --------------------------------------------------------------------------- #
@pytest.fixture
def pkl_dataset_dir():
    """A tiny synthetic .pkl dataset in the OnHW-chars_L layout."""
    rng = np.random.default_rng(0)
    alphabet = "ABCabc"  # 6 classes for a fast fixture
    writers_raw = [10, 10, 10, 20, 20, 20, 30, 30, 30]  # 3 writers, non-contiguous IDs
    n = len(writers_raw)
    x = [rng.normal(0, 1, size=(20 + i % 3, 13)).astype(np.float32) for i in range(n)]
    y_str = list(alphabet) * (n // len(alphabet))
    y_int = list(range(len(alphabet))) * (n // len(alphabet))

    d = tempfile.mkdtemp(prefix="onhw_chars_pkl_")
    for fname, obj in [
        ("all_x_dat_imu.pkl", x),
        ("all_gt.pkl", y_str),
        ("all_gt_enc.pkl", y_int),
        ("list_ids.pkl", writers_raw),
    ]:
        with open(os.path.join(d, fname), "wb") as f:
            pickle.dump(obj, f)
    yield d
    shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Synthetic .npy dataset (mirrors the OnHW-chars right-handed layout)
# --------------------------------------------------------------------------- #
@pytest.fixture
def npy_dataset_dir():
    """A tiny synthetic .npy dataset in the OnHW-chars right-handed layout."""
    rng = np.random.default_rng(0)
    alphabet = "ABCabc"
    n_classes = len(alphabet)
    d = tempfile.mkdtemp(prefix="onhw_chars_npy_")
    for case in ["lower", "upper", "both"]:
        for dep in ["dep", "indep"]:
            for fold in range(5):
                folder = os.path.join(d, f"onhw2_{case}_{dep}_{fold}")
                os.makedirs(folder, exist_ok=True)
                # Train: 6 samples, Test: 3 samples
                X_train = np.array(
                    [
                        rng.normal(0, 1, size=(20 + i, 13)).astype(np.float32)
                        for i in range(6)
                    ],
                    dtype=object,
                )
                X_test = np.array(
                    [
                        rng.normal(0, 1, size=(20 + i, 13)).astype(np.float32)
                        for i in range(3)
                    ],
                    dtype=object,
                )
                y_train = np.array([i % n_classes for i in range(6)], dtype=np.int64)
                y_test = np.array([i % n_classes for i in range(3)], dtype=np.int64)
                np.save(os.path.join(folder, "X_train.npy"), X_train, allow_pickle=True)
                np.save(os.path.join(folder, "X_test.npy"), X_test, allow_pickle=True)
                np.save(os.path.join(folder, "y_train.npy"), y_train, allow_pickle=True)
                np.save(os.path.join(folder, "y_test.npy"), y_test, allow_pickle=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# .pkl format tests
# --------------------------------------------------------------------------- #
def test_load_pkl_basic(pkl_dataset_dir):
    ds = C.load_onhw_chars(pkl_dataset_dir)
    assert ds.format == "pkl"
    assert ds.n_samples == 9
    assert ds.n_classes == 6
    assert ds.n_writers == 3  # 3 unique raw IDs remapped to 0,1,2
    assert ds.classes == sorted("ABCabc")
    # X_all is a list of (T, 13) float arrays
    assert all(isinstance(s, np.ndarray) for s in ds.X_all)
    assert all(s.shape[1] == 13 for s in ds.X_all)
    # No splits in .pkl format
    assert ds.X_train is None
    assert ds.X_test is None


def test_pkl_remaps_non_contiguous_writer_ids(pkl_dataset_dir):
    """Fraunhofer uses non-contiguous recording IDs (e.g. [10, 20, 30]); the
    loader must remap them to 0..N-1 so downstream code can use them as
    array indices."""
    ds = C.load_onhw_chars(pkl_dataset_dir)
    assert set(ds.writers.tolist()) == {0, 1, 2}
    assert ds.writers.max() == 2


def test_pkl_label_encoding_is_canonical(pkl_dataset_dir):
    """Labels must be encoded in alphabetical order (so char_to_idx matches
    onhw_models.load_raw)."""
    ds = C.load_onhw_chars(pkl_dataset_dir)
    # classes is sorted alphabetically
    assert ds.classes == ["A", "B", "C", "a", "b", "c"]
    # first sample's label corresponds to first character in the alphabet cycle
    assert ds.y_all[0] == 0  # 'A'
    assert ds.y_all[3] == 3  # 'a'


def test_pkl_summary_string(pkl_dataset_dir):
    ds = C.load_onhw_chars(pkl_dataset_dir)
    s = ds.summary()
    assert "pkl" in s and "N=9" in s and "C=6" in s and "W=3" in s


# --------------------------------------------------------------------------- #
# .npy format tests
# --------------------------------------------------------------------------- #
def test_load_npy_basic(npy_dataset_dir):
    ds = C.load_onhw_chars(npy_dataset_dir, case="both", dependency="indep", fold=0)
    assert ds.format == "npy"
    assert ds.X_train is not None
    assert ds.X_test is not None
    assert len(ds.X_train) == 6
    assert len(ds.X_test) == 3
    assert ds.n_samples == 9
    # 'both' case = 52 classes (the loader uses the canonical OnHW class set
    # regardless of how many classes the synthetic .npy mock has - the real
    # dataset always has 52 for 'both').
    assert ds.n_classes == 52
    assert ds.classes == list(C.CHARS_BOTH)


def test_npy_case_upper(npy_dataset_dir):
    ds = C.load_onhw_chars(npy_dataset_dir, case="upper", dependency="dep", fold=2)
    assert ds.format == "npy"
    assert ds.classes == list(C.CHARS_UPPER)


def test_npy_case_lower(npy_dataset_dir):
    ds = C.load_onhw_chars(npy_dataset_dir, case="lower", dependency="indep", fold=4)
    assert ds.format == "npy"
    assert ds.classes == list(C.CHARS_LOWER)


def test_npy_invalid_case_raises(npy_dataset_dir):
    with pytest.raises(ValueError):
        C.load_onhw_chars(npy_dataset_dir, case="mixed", dependency="indep", fold=0)


def test_npy_invalid_fold_raises(npy_dataset_dir):
    with pytest.raises(ValueError):
        C.load_onhw_chars(npy_dataset_dir, case="both", dependency="indep", fold=5)


def test_npy_missing_folder_raises(tmp_path):
    # tmp_path exists but has no onhw2_* subfolder and no all_x_dat_imu.pkl
    with pytest.raises(FileNotFoundError, match="could not detect"):
        C.load_onhw_chars(str(tmp_path))


# --------------------------------------------------------------------------- #
# Constants and channel layout
# --------------------------------------------------------------------------- #
def test_channel_layout():
    assert len(C.CHANNEL_NAMES) == 13
    assert C.CHANNEL_NAMES[12] == "force"
    assert C.N_CHANNELS == 13


def test_class_strings():
    assert len(C.CHARS_BOTH) == 52
    assert len(C.CHARS_UPPER) == 26
    assert len(C.CHARS_LOWER) == 26
    assert C.CHARS_UPPER[0] == "A"
    assert C.CHARS_LOWER[0] == "a"


# --------------------------------------------------------------------------- #
# Real OnHW-chars_L dataset (only if downloaded)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not os.path.exists(
        "/home/z/my-project/imu2text-work/onhw_data/OnHW-chars_L/OnHW-chars_L/all_x_dat_imu.pkl"
    ),
    reason="OnHW-chars_L not downloaded (smoke-test only)",
)
def test_real_onhw_chars_L_loads():
    """Smoke test against the real Fraunhofer OnHW-chars_L dataset."""
    ds = C.load_onhw_chars(
        "/home/z/my-project/imu2text-work/onhw_data/OnHW-chars_L/OnHW-chars_L"
    )
    assert ds.format == "pkl"
    assert ds.n_samples == 2270
    assert ds.n_classes == 52
    assert ds.n_writers == 9
    assert ds.classes[0] == "A"
    assert ds.classes[25] == "Z"
    assert ds.classes[26] == "a"
    assert ds.classes[51] == "z"
    # All sequences have 13 channels
    assert all(s.shape[1] == 13 for s in ds.X_all)
    # Writer IDs are remapped to 0..8
    assert set(ds.writers.tolist()) == set(range(9))


# --------------------------------------------------------------------------- #
# Writer IDs
#
# The .npy archives ship pre-made splits and no per-sample writer IDs. Filling
# those in with zeros would claim every sample came from one writer, which
# reads as valid data downstream and turns a writer-independent re-split into
# a silently leaky one. They are marked unknown instead.
# --------------------------------------------------------------------------- #
def test_npy_writer_ids_are_marked_unknown(npy_dataset_dir):
    ds = C.load_onhw_chars(npy_dataset_dir, case="both", dependency="indep", fold=0)
    assert np.all(ds.writers == C.WRITER_UNKNOWN)
    assert ds.has_writer_ids is False


def test_npy_n_writers_is_zero_rather_than_a_made_up_count(npy_dataset_dir):
    ds = C.load_onhw_chars(npy_dataset_dir, case="both", dependency="indep", fold=0)
    assert ds.n_writers == 0


def test_pkl_writer_ids_are_real(pkl_dataset_dir):
    ds = C.load_onhw_chars(pkl_dataset_dir)
    assert ds.has_writer_ids is True
    assert np.all(ds.writers >= 0)
    assert ds.n_writers > 0


def test_writer_unknown_sentinel_is_negative():
    """make_split keys off the sign, so a non-negative sentinel would slip past."""
    assert C.WRITER_UNKNOWN < 0


def test_unknown_writer_ids_are_refused_by_the_writer_split(npy_dataset_dir):
    """The two halves of the guard have to line up: sentinel in, error out."""
    pytest.importorskip("tensorflow")
    import onhw_models as M

    ds = C.load_onhw_chars(npy_dataset_dir, case="both", dependency="indep", fold=0)
    with pytest.raises(ValueError, match="unknown"):
        M.make_split(len(ds.X_all), ds.y_all, seed=0, mode="writer", writers=ds.writers)


def test_remap_writer_ids_makes_them_contiguous():
    """The shipped left-handed IDs are recording numbers like 1052 and 4003."""
    raw = np.array([0, 1, 2, 3, 4, 5, 6, 1052, 4003])
    out = C._remap_writer_ids(raw)
    assert out.tolist() == list(range(9))


def test_remap_writer_ids_preserves_grouping():
    raw = np.array([4003, 1052, 4003, 0, 1052, 0])
    out = C._remap_writer_ids(raw)
    for a, raw_a in enumerate(raw):
        for b, raw_b in enumerate(raw):
            assert (raw_a == raw_b) == (out[a] == out[b])
