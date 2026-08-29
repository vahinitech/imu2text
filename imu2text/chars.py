"""OnHW-chars dataset loader (both .npy and .pkl formats).

The Fraunhofer IIS OnHW-chars dataset is published in two formats:

1. **Right-handed (.npy)** - 31,275 samples from 119 writers, 52 classes
   (A-Z, a-z). Ships with 30 pre-computed train/test splits. The archive
   unpacks to a lowercase ``onhw-chars_2021-06-30/`` folder (verified
   against the published ZIP's directory listing):

       onhw-chars_2021-06-30/
       ├── onhw2_lower_dep_0/        # case x dependency x fold
       │   ├── X_train.npy
       │   ├── X_test.npy
       │   ├── y_train.npy
       │   └── y_test.npy
       ├── onhw2_lower_dep_1/
       ├── ...
       ├── onhw2_both_indep_4/
       └── readme.txt

   - ``case``       : ``lower`` (a-z, 26 classes), ``upper`` (A-Z, 26), or
                      ``both`` (A-Z + a-z, 52 classes).
   - ``dependency`` : ``dep`` (writer-dependent, same writers in train/test)
                      or ``indep`` (writer-independent, disjoint writers).
   - ``fold``       : 0-4 (5-fold cross validation).

   ``X_train`` / ``X_test`` are object arrays of ``(T, 13)`` float arrays
   (variable T). ``y_train`` / ``y_test`` are int arrays (0..25 for lower/
   upper, 0..51 for both).

2. **Left-handed (.pkl)** - 2,270 samples from 9 writers, 52 classes. No
   official splits; ships as four pickles:

       OnHW-chars_L/
       ├── all_x_dat_imu.pkl    # list[np.ndarray (T, 13)]
       ├── all_gt.pkl           # list[str] of length N (e.g. ['A','B','c',...])
       ├── all_gt_enc.pkl       # list[int] of length N (0..51)
       └── list_ids.pkl         # list[int] of writer IDs

   Note: writer IDs in ``list_ids.pkl`` use the original Fraunhofer recording
   IDs, which are *not* zero-indexed (the 9 writers have IDs like
   ``[0,1,2,3,4,5,6,1052,4003]``). ``load_chars_pkl`` remaps these to a
   contiguous ``0..num_writers-1`` range for downstream use.

This module exposes a single ``load_onhw_chars`` entry point that detects
the format from the directory contents and returns a unified
``OnHWCharsDataset`` named tuple.

Channel layout (13 channels, identical across both formats):

    [0:3]   Acc1 X, Y, Z      front accelerometer
    [3:6]   Acc2 X, Y, Z      rear accelerometer
    [6:9]   Gyro  X, Y, Z     gyroscope
    [9:12]  Mag   X, Y, Z     magnetometer
    [12]    Force             pen-tip force

Usage
-----
    # .npy format (right-handed, with official 5-fold splits)
    ds = load_onhw_chars("./data/onhw-chars_2021-06-30",
                         case="both", dependency="indep", fold=0)
    X_train, y_train = ds.X_train, ds.y_train

    # .pkl format (left-handed, no splits - infer writers and split yourself)
    ds = load_onhw_chars("./data/OnHW-chars_L")
    x, y, writers = ds.X_all, ds.y_all, ds.writers
"""

from __future__ import annotations

import os
import pickle
from typing import List, NamedTuple, Optional

import numpy as np

# Channel layout - shared with imu2text.augment.SENSOR_GROUPS but kept local to
# avoid a circular import (onhw_augment is the canonical home, this is just
# documentation of what each column is for downstream users).
CHANNEL_NAMES = [
    "acc1_x",
    "acc1_y",
    "acc1_z",  # 0-2: front accelerometer
    "acc2_x",
    "acc2_y",
    "acc2_z",  # 3-5: rear accelerometer
    "gyro_x",
    "gyro_y",
    "gyro_z",  # 6-8: gyroscope
    "mag_x",
    "mag_y",
    "mag_z",  # 9-11: magnetometer
    "force",  # 12: pen-tip force
]
N_CHANNELS = 13

#: Sentinel writer ID for samples whose writer is not recorded in the archive
#: (the .npy OnHW-chars release ships splits but no per-sample writer IDs).
WRITER_UNKNOWN = -1

# Canonical 52-class label set: A-Z then a-z. The .npy splits use integer
# labels in this exact order (so label 0 == 'A', 25 == 'Z', 26 == 'a', 51 == 'z').
CHARS_BOTH = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHARS_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS_LOWER = "abcdefghijklmnopqrstuvwxyz"


class OnHWCharsDataset(NamedTuple):
    """Unified container for OnHW-chars data, regardless of source format.

    The .npy format has explicit train/test splits, so ``X_train`` / ``y_train``
    / ``X_test`` / ``y_test`` are populated and ``X_all`` / ``y_all`` / \
    ``writers`` are derived by concatenation. The .pkl format has no splits,
    so ``X_all`` / ``y_all`` / ``writers`` are populated and the train/test
    arrays are ``None`` (the caller is expected to split using
    ``imu2text.models.make_split`` with the inferred writer IDs).
    """

    X_train: Optional[List[np.ndarray]]
    y_train: Optional[np.ndarray]
    X_test: Optional[List[np.ndarray]]
    y_test: Optional[np.ndarray]
    X_all: List[np.ndarray]
    y_all: np.ndarray
    writers: np.ndarray  # per-sample writer IDs (0-indexed),
    # or all WRITER_UNKNOWN for .npy splits
    classes: List[str]  # e.g. list("ABC...Zabc...z")
    format: str  # "npy" or "pkl"

    @property
    def n_samples(self) -> int:
        return len(self.X_all)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_writers(self) -> int:
        """Number of distinct writers, or 0 when the IDs are unknown."""
        known = self.writers[self.writers != WRITER_UNKNOWN]
        return int(known.max()) + 1 if len(known) else 0

    @property
    def has_writer_ids(self) -> bool:
        """True when per-sample writer IDs are available (the .pkl format).

        The .npy archives ship pre-made splits and no writer IDs, so a
        writer-independent re-split is not possible from them - use their
        ``dependency="indep"`` folds instead.
        """
        return bool(np.any(self.writers != WRITER_UNKNOWN))

    def summary(self) -> str:
        lens = [len(s) for s in self.X_all]
        return (
            f"OnHW-chars ({self.format}): N={self.n_samples}, "
            f"C={self.n_classes}, W={self.n_writers}, "
            f"len mean={np.mean(lens):.1f} min={min(lens)} max={max(lens)}"
        )


# --------------------------------------------------------------------------- #
# .npy loader (right-handed, 30 official splits)
# --------------------------------------------------------------------------- #
def _load_npy_split(base: str, case: str, dependency: str, fold: int):
    """Load one of the 30 official .npy splits.

    Returns (X_train, y_train, X_test, y_test, classes).
    """
    if case not in ("lower", "upper", "both"):
        raise ValueError(f"case must be lower/upper/both, got {case!r}")
    if dependency not in ("dep", "indep"):
        raise ValueError(f"dependency must be dep/indep, got {dependency!r}")
    if not 0 <= fold <= 4:
        raise ValueError(f"fold must be 0-4, got {fold}")

    folder = os.path.join(base, f"onhw2_{case}_{dependency}_{fold}")
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"split folder not found: {folder}. Did you download onhw_chars?"
        )

    X_train = list(np.load(os.path.join(folder, "X_train.npy"), allow_pickle=True))
    X_test = list(np.load(os.path.join(folder, "X_test.npy"), allow_pickle=True))
    y_train = np.load(os.path.join(folder, "y_train.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(folder, "y_test.npy"), allow_pickle=True)
    classes = list(
        {"lower": CHARS_LOWER, "upper": CHARS_UPPER, "both": CHARS_BOTH}[case]
    )
    y_train = _encode_labels(y_train, classes, folder)
    y_test = _encode_labels(y_test, classes, folder)
    return X_train, y_train, X_test, y_test, classes


def _encode_labels(y: np.ndarray, classes: List[str], folder: str) -> np.ndarray:
    """Return integer class indices, whatever the archive stored.

    The published .npy splits store labels as single-character strings
    (``dtype='<U1'``: ``['A', 'B', ...]``), not integers. They are mapped to
    the canonical class order so an index means the same character here as it
    does everywhere else in the repo. Integer arrays are passed through, so a
    re-encoded copy of the dataset still loads.
    """
    y = np.asarray(y)
    if y.dtype.kind in ("i", "u"):
        return y.astype(np.int64)
    char_to_idx = {c: i for i, c in enumerate(classes)}
    try:
        return np.array([char_to_idx[str(v)] for v in y], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(
            f"label {exc.args[0]!r} in {folder} is not in the {len(classes)}-class "
            f"set for this case ({''.join(classes)!r}). Wrong --case?"
        ) from None


# --------------------------------------------------------------------------- #
# .pkl loader (left-handed, no splits)
# --------------------------------------------------------------------------- #
def _load_pkl_chars(base: str):
    """Load the four .pkl files of the left-handed OnHW-chars dataset.

    Returns (X_all, y_str_all, y_int_all, writer_ids_raw). Writer IDs are
    returned as-is (they use Fraunhofer's original recording IDs, which are
    not zero-indexed); the caller is expected to remap them.
    """
    required = ["all_x_dat_imu.pkl", "all_gt.pkl", "all_gt_enc.pkl", "list_ids.pkl"]
    for f in required:
        if not os.path.exists(os.path.join(base, f)):
            raise FileNotFoundError(
                f"missing {f} in {base}. Did you download onhw_chars_L?"
            )

    with open(os.path.join(base, "all_x_dat_imu.pkl"), "rb") as f:
        X = [np.asarray(s, dtype=np.float32) for s in pickle.load(f)]
    with open(os.path.join(base, "all_gt.pkl"), "rb") as f:
        y_str = list(pickle.load(f))
    with open(os.path.join(base, "all_gt_enc.pkl"), "rb") as f:
        y_int = np.array(list(pickle.load(f)), dtype=np.int64)
    with open(os.path.join(base, "list_ids.pkl"), "rb") as f:
        writers_raw = np.array(list(pickle.load(f)), dtype=np.int64)
    return X, y_str, y_int, writers_raw


def _remap_writer_ids(writers_raw: np.ndarray) -> np.ndarray:
    """Remap arbitrary writer IDs to a contiguous 0..N-1 range.

    Fraunhofer's .pkl datasets use original recording IDs (e.g.
    ``[0,1,2,3,4,5,6,1052,4003]`` for 9 writers) which break downstream
    code that assumes writer IDs are array indices. We map each unique ID
    to its rank in sorted order, preserving the original writer count.
    """
    uniq = np.unique(writers_raw)
    remap = {raw: i for i, raw in enumerate(uniq)}
    return np.array([remap[w] for w in writers_raw], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_onhw_chars(
    base_dir: str, case: str = "both", dependency: str = "indep", fold: int = 0
) -> OnHWCharsDataset:
    """Load OnHW-chars from either format, auto-detected from ``base_dir``.

    Parameters
    ----------
    base_dir : str
        Path to the extracted OnHW-chars folder. The loader auto-detects:

        - **.npy format** (right-handed): ``base_dir`` contains
          ``onhw2_{case}_{dependency}_{fold}/`` subfolders.
        - **.pkl format** (left-handed): ``base_dir`` contains
          ``all_x_dat_imu.pkl`` etc. The ``case``/``dependency``/``fold``
          arguments are ignored (the .pkl release has no official splits).

    case : str, default "both"
        Character case for the .npy format: ``lower`` (26 classes), ``upper``
        (26 classes), or ``both`` (52 classes, the standard OnHW benchmark).
    dependency : str, default "indep"
        Split type for the .npy format: ``dep`` (writer-dependent) or
        ``indep`` (writer-independent, the protocol the OnHW papers report).
    fold : int, default 0
        Fold index (0-4) for the .npy format.

    Returns
    -------
    OnHWCharsDataset
        Named tuple with both per-split (X_train, y_train, X_test, y_test)
        and aggregated (X_all, y_all, writers) views.
    """
    # Validate arguments first so an invalid case/dependency/fold always raises
    # ValueError, even if the directory doesn't contain the corresponding
    # .npy split folder.
    if case not in ("lower", "upper", "both"):
        raise ValueError(f"case must be lower/upper/both, got {case!r}")
    if dependency not in ("dep", "indep"):
        raise ValueError(f"dependency must be dep/indep, got {dependency!r}")
    if not 0 <= fold <= 4:
        raise ValueError(f"fold must be 0-4, got {fold}")

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"directory not found: {base_dir}")

    # Auto-detect: if any onhw2_* subfolder exists, it's the .npy format.
    npy_marker = os.path.join(base_dir, f"onhw2_{case}_{dependency}_{fold}")
    pkl_marker = os.path.join(base_dir, "all_x_dat_imu.pkl")

    if os.path.isdir(npy_marker):
        X_train, y_train, X_test, y_test, classes = _load_npy_split(
            base_dir, case, dependency, fold
        )
        # The .npy splits bake the writer partition into the split itself and
        # ship no per-sample writer IDs. Filling in zeros would claim "one
        # writer" and quietly turn a writer-independent re-split into a
        # random one, so mark them unknown with -1 instead - callers can test
        # for it, and make_split(mode="writer") will fail loudly rather than
        # silently produce a leaky split. Use the official X_train/X_test
        # split for these archives.
        X_all = list(X_train) + list(X_test)
        y_all = np.concatenate([y_train, y_test]).astype(np.int64)
        writers = np.full(len(X_all), WRITER_UNKNOWN, dtype=np.int64)
        return OnHWCharsDataset(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_all=X_all,
            y_all=y_all,
            writers=writers,
            classes=classes,
            format="npy",
        )

    if os.path.exists(pkl_marker):
        # The stored integer encoding is discarded: labels are re-derived from
        # the strings below so the class order matches load_raw's everywhere.
        X_all, y_str, _stored_enc, writers_raw = _load_pkl_chars(base_dir)
        writers = _remap_writer_ids(writers_raw)
        # Use the string labels for the class list so the order is the same
        # as onhw_models.load_raw (alphabetical), independent of Fraunhofer's
        # int encoding.
        classes = sorted(set(y_str))
        # Re-encode labels in our canonical (alphabetical) order so they're
        # consistent with onhw_models.load_raw's char_to_idx.
        char_to_idx = {c: i for i, c in enumerate(classes)}
        y_all = np.array([char_to_idx[c] for c in y_str], dtype=np.int64)
        return OnHWCharsDataset(
            X_train=None,
            y_train=None,
            X_test=None,
            y_test=None,
            X_all=X_all,
            y_all=y_all,
            writers=writers,
            classes=classes,
            format="pkl",
        )

    raise FileNotFoundError(
        f"could not detect OnHW-chars format in {base_dir}. Expected either "
        f"an `onhw2_*` subfolder (.npy) or `all_x_dat_imu.pkl` (.pkl). "
        "Did you extract the ZIP archive?"
    )


# --------------------------------------------------------------------------- #
# CLI for quick inspection
# --------------------------------------------------------------------------- #
def main() -> None:
    """CLI: load an OnHW-chars folder and print what it contains."""
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base_dir", help="path to the extracted OnHW-chars folder")
    ap.add_argument(
        "--case",
        default="both",
        choices=["lower", "upper", "both"],
        help="character case (.npy only; default: both)",
    )
    ap.add_argument(
        "--dependency",
        default="indep",
        choices=["dep", "indep"],
        help="split type (.npy only; default: indep)",
    )
    ap.add_argument(
        "--fold", type=int, default=0, help="fold index 0-4 (.npy only; default: 0)"
    )
    args = ap.parse_args()

    ds = load_onhw_chars(args.base_dir, args.case, args.dependency, args.fold)
    print(ds.summary())
    print(f"Format: {ds.format}")
    print(f"Classes ({ds.n_classes}): {''.join(ds.classes)}")
    if ds.X_train is not None:
        print(f"Train: {len(ds.X_train)} samples, Test: {len(ds.X_test)} samples")
        print(
            f"Train label balance: min={np.bincount(ds.y_train).min()}, "
            f"max={np.bincount(ds.y_train).max()}"
        )
    else:
        print(f"Writers (remapped): {sorted(set(ds.writers.tolist()))}")
        print(f"Samples per writer: {np.bincount(ds.writers).tolist()}")


if __name__ == "__main__":
    main()
