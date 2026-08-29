"""OnHW-symbols and OnHW-equations dataset loaders + transfer learning.

This module covers the two single-charset OnHW datasets that share the same
15-class symbol vocabulary (digits 0-9 and operators + - · : =):

- **OnHW-symbols** - single-symbol classification. 2,326 samples, 27 writers.
  Each sample is one IMU recording of one symbol being written.
- **OnHW-equations** - sequence-to-sequence recognition. 10,713 samples, 55
  writers. Each sample is a recording of an equation like "12+3=15", labeled
  with the sequence of symbols.

Both ship in the same flat .pkl layout with one official train/val split
per archive (dep/indep variants). The
file suffixes distinguish them:

    OnHW-symbols_equations_dep/          # and _indep; verified 2021-09-02
    ├── all_x_dat_train_imu_s.pkl        # _s = symbols (one symbol per sample)
    ├── all_x_dat_val_imu_s.pkl
    ├── all_train_gt_s.pkl               # list[int], 0..14
    ├── all_val_gt_s.pkl
    ├── train_ids_s.pkl                  # writer IDs
    ├── val_ids_s.pkl
    ├── ... the same six files with an _e suffix (equations, per-symbol
    │       slices) and an _es suffix (equations re-sliced into symbols)
    ├── all_indices_e.txt                # source recording per equation slice
    └── all_symbol_lens_e.txt            # per-writer length statistics

There are no ``fold_*`` subfolders: each archive ships exactly one official
train/val split (1853/473 samples for ``dep``), and that split is what the
loader returns. ``dep`` shares all 27 writers between train and val;
``indep`` keeps them disjoint. Deriving a different split here would
silently change which protocol a reported number belongs to, so the shipped
one is used verbatim - check ``ds.is_writer_independent`` before labelling
a result.

The left-handed variant (OnHW-symbols_equations_L) ships no split at all:

    OnHW-symbols_equations_L/
    ├── all_x_dat_imu_s.pkl              # everything in one set
    ├── all_gt_s.pkl
    ├── list_ids_s.pkl
    └── ... the same three with an _e suffix

It loads with ``has_official_split == False``, everything in ``X_train``
and an empty val - split it yourself before evaluating anything.

Channel layout is the same 13-channel OnHW pen format as the chars dataset.
The 15-class symbol charset is:

    "0123456789+-·:="

(indices 0-9 are digits, 10-14 are operators).

Transfer learning
-----------------
OnHW-symbols is tiny (~2.3k samples, ~27 writers), so training from scratch
is wasteful when OnHW-chars (31k samples, 119 writers) is available. This
module exposes a ``build_transfer_model`` helper that:

1. Loads a pretrained CNN+BiLSTM model trained on OnHW-chars (the conv trunk
   has already learned useful local stroke features).
2. Removes the original 52-class classification head.
3. Attaches a new 15-class softmax head.
4. Freezes the conv trunk for a few warmup epochs (only the new head trains),
   then unfreezes everything for fine-tuning at a low learning rate.

This is the standard transfer-learning recipe and typically dominates
training from scratch on OnHW-symbols by 5-10 points.

Usage
-----
    from onhw_symbols import load_onhw_symbols, SYMBOLS_VOCAB

    # Load symbols (single-symbol classification)
    ds = load_onhw_symbols("./data/OnHW-symbols_equations_dep")
    X_train, y_train = ds.X_train, ds.y_train

    # Load equations (sequence-to-sequence)
    ds = load_onhw_equations("./data/OnHW-equations_dep")
    # Use onhw_seq2seq for CTC training on these
"""

from __future__ import annotations

import os
import pickle
from typing import List, NamedTuple, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Symbol vocabulary (15 classes, shared by symbols and equations)
# --------------------------------------------------------------------------- #
# 0-9 are digits, 10-14 are operators: + - · : =
SYMBOLS_VOCAB = "0123456789+-·:="
SYMBOLS_BLANK_IDX = len(SYMBOLS_VOCAB)  # CTC blank = index 15


def encode_symbol(s: str) -> int:
    """Encode a single symbol character to its integer index (0..14)."""
    idx = SYMBOLS_VOCAB.find(s)
    if idx < 0:
        raise ValueError(f"unknown symbol {s!r}; must be one of {SYMBOLS_VOCAB!r}")
    return idx


def decode_symbol(idx: int) -> str:
    """Decode an integer index (0..14) back to a symbol character."""
    if not 0 <= idx < len(SYMBOLS_VOCAB):
        raise ValueError(f"index {idx} out of range [0, {len(SYMBOLS_VOCAB)})")
    return SYMBOLS_VOCAB[idx]


def decode_symbol_seq(indices: Sequence[int]) -> str:
    """Decode a sequence of integer indices to a string."""
    return "".join(decode_symbol(i) for i in indices if 0 <= i < len(SYMBOLS_VOCAB))


# --------------------------------------------------------------------------- #
# Dataset containers
# --------------------------------------------------------------------------- #
class OnHWSymbolsDataset(NamedTuple):
    """OnHW-symbols single-symbol classification dataset (15 classes)."""

    X_train: List[np.ndarray]
    X_val: List[np.ndarray]
    y_train: np.ndarray  # int64, 0..14
    y_val: np.ndarray
    train_ids: np.ndarray
    val_ids: np.ndarray
    split: str = "official"  # "official" (shipped) or "none" (unsplit)
    format: str = "symbols_pkl"

    @property
    def has_official_split(self) -> bool:
        """True when the archive shipped its own train/val split."""
        return self.split == "official"

    @property
    def is_writer_independent(self) -> bool:
        """True when no writer appears in both train and val.

        The archive name says which protocol it intends (``dep`` shares
        writers, ``indep`` does not); this checks what the data actually
        does, so a number can be labelled from the data rather than the
        filename. Meaningless when there is no val split.
        """
        if not len(self.val_ids):
            return False
        return set(self.train_ids.tolist()).isdisjoint(set(self.val_ids.tolist()))

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_val(self) -> int:
        return len(self.X_val)

    @property
    def n_classes(self) -> int:
        return len(SYMBOLS_VOCAB)

    @property
    def n_writers(self) -> int:
        return len(set(self.train_ids.tolist() + self.val_ids.tolist()))

    def summary(self) -> str:
        lens = [len(s) for s in self.X_train] + [len(s) for s in self.X_val]
        protocol = (
            "no split shipped"
            if not self.has_official_split
            else (
                "writer-independent"
                if self.is_writer_independent
                else "writer-dependent"
            )
        )
        return (
            f"OnHW-symbols ({protocol}): "
            f"train={self.n_train} val={self.n_val} "
            f"writers={self.n_writers} classes={self.n_classes} "
            f"len mean={np.mean(lens):.0f} max={max(lens)}"
        )


class OnHWEquationsDataset(NamedTuple):
    """OnHW-equations sequence-to-sequence dataset (15-symbol charset)."""

    X_train: List[np.ndarray]
    X_val: List[np.ndarray]
    Y_train: List[List[int]]  # token sequences, each 0..14
    Y_val: List[List[int]]
    train_words: List[str]  # decoded equation strings
    val_words: List[str]
    train_ids: np.ndarray
    val_ids: np.ndarray
    split: str = "official"  # "official" (shipped) or "none" (unsplit)
    format: str = "equations_pkl"

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_val(self) -> int:
        return len(self.X_val)

    @property
    def n_writers(self) -> int:
        return len(set(self.train_ids.tolist() + self.val_ids.tolist()))

    @property
    def vocab_size(self) -> int:
        return len(SYMBOLS_VOCAB)

    @property
    def lexicon(self) -> List[str]:
        """Sorted unique equation strings across train+val."""
        return sorted(set(self.train_words + self.val_words))

    def summary(self) -> str:
        lens = [len(s) for s in self.X_train] + [len(s) for s in self.X_val]
        label_lens = [len(seq) for seq in self.Y_train + self.Y_val]
        protocol = (
            "no split shipped"
            if self.split != "official"
            else (
                "writer-independent"
                if len(self.val_ids)
                and set(self.train_ids.tolist()).isdisjoint(set(self.val_ids.tolist()))
                else "writer-dependent"
            )
        )
        return (
            f"OnHW-equations ({protocol}): "
            f"train={self.n_train} val={self.n_val} "
            f"writers={self.n_writers} "
            f"lexicon={len(self.lexicon)} eqs "
            f"IMU len mean={np.mean(lens):.0f} "
            f"label len mean={np.mean(label_lens):.1f}"
        )


# --------------------------------------------------------------------------- #
# Helpers - locate and load the per-fold pkl files
# --------------------------------------------------------------------------- #
def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _official_split_paths(base_dir: str, suffix: str):
    """Paths for the shipped train/val split, or None if the archive has none.

    The dep/indep archives are flat and carry an official split:

        all_x_dat_train_imu_{suffix}.pkl   all_x_dat_val_imu_{suffix}.pkl
        all_train_gt_{suffix}.pkl          all_val_gt_{suffix}.pkl
        train_ids_{suffix}.pkl             val_ids_{suffix}.pkl
    """
    names = {
        "X_train": f"all_x_dat_train_imu_{suffix}.pkl",
        "X_val": f"all_x_dat_val_imu_{suffix}.pkl",
        "y_train": f"all_train_gt_{suffix}.pkl",
        "y_val": f"all_val_gt_{suffix}.pkl",
        "train_ids": f"train_ids_{suffix}.pkl",
        "val_ids": f"val_ids_{suffix}.pkl",
    }
    paths = {k: os.path.join(base_dir, v) for k, v in names.items()}
    return paths if all(os.path.exists(p) for p in paths.values()) else None


def _flat_paths(base_dir: str, suffix: str):
    """Paths for the unsplit left-handed layout, or None if it isn't that."""
    names = {
        "X_all": f"all_x_dat_imu_{suffix}.pkl",
        "y_all": f"all_gt_{suffix}.pkl",
        "ids": f"list_ids_{suffix}.pkl",
    }
    paths = {k: os.path.join(base_dir, v) for k, v in names.items()}
    return paths if all(os.path.exists(p) for p in paths.values()) else None


def _load_symbol_arrays(base_dir: str, suffix: str, kind: str):
    """Load one sub-dataset, preferring the archive's own train/val split.

    Returns ``(X_train, y_train, train_ids, X_val, y_val, val_ids, split)``
    where ``split`` is ``"official"`` or ``"none"``.

    The published splits are used verbatim rather than re-derived. The
    ``dep`` archives deliberately share writers between train and val and
    the ``indep`` ones deliberately do not; synthesising a writer-disjoint
    split here would turn the writer-dependent archive into a
    writer-independent evaluation and quietly mislabel whatever protocol
    the resulting number was measured under.
    """
    official = _official_split_paths(base_dir, suffix)
    if official is not None:
        return (
            [np.asarray(s, dtype=np.float32) for s in _load_pkl(official["X_train"])],
            np.array(list(_load_pkl(official["y_train"])), dtype=np.int64),
            np.array(list(_load_pkl(official["train_ids"])), dtype=np.int64),
            [np.asarray(s, dtype=np.float32) for s in _load_pkl(official["X_val"])],
            np.array(list(_load_pkl(official["y_val"])), dtype=np.int64),
            np.array(list(_load_pkl(official["val_ids"])), dtype=np.int64),
            "official",
        )

    flat = _flat_paths(base_dir, suffix)
    if flat is not None:
        empty_x: List[np.ndarray] = []
        return (
            [np.asarray(s, dtype=np.float32) for s in _load_pkl(flat["X_all"])],
            np.array(list(_load_pkl(flat["y_all"])), dtype=np.int64),
            np.array(list(_load_pkl(flat["ids"])), dtype=np.int64),
            empty_x,
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            "none",
        )

    raise FileNotFoundError(
        f"no OnHW-{kind} pickles found in {base_dir}. Expected either the "
        f"split layout (all_x_dat_train_imu_{suffix}.pkl + "
        f"all_x_dat_val_imu_{suffix}.pkl + ...) shipped by the dep/indep "
        f"archives, or the unsplit layout (all_x_dat_imu_{suffix}.pkl + "
        f"all_gt_{suffix}.pkl + list_ids_{suffix}.pkl) shipped by the "
        "left-handed archive. Download with "
        "`python onhw_download.py onhw_symbols_dep` (or onhw_symbols_L)."
    )


def load_onhw_symbols(base_dir: str) -> OnHWSymbolsDataset:
    """Load the OnHW-symbols single-symbol classification dataset (15 classes).

    Parameters
    ----------
    base_dir : str
        Path to an extracted archive folder, e.g.
        ``OnHW-symbols_equations_dep`` or ``OnHW-symbols_equations_L``.

    The dep/indep archives ship one official train/val split (1853/473 for
    ``dep``), which is used as-is. Which protocol that split represents is
    the archive's choice, not this loader's: ``dep`` shares all 27 writers
    between train and val, ``indep`` keeps them disjoint. Check
    ``is_writer_independent`` before labelling any number you report.

    The left-handed archive ships no split at all. Everything is returned as
    train with an empty val, and ``has_official_split`` is False - build a
    split yourself (``onhw_models.make_split(mode="writer",
    writers=ds.train_ids)``) rather than evaluating on the empty val.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"directory not found: {base_dir}")
    Xtr, ytr, itr, Xva, yva, iva, split = _load_symbol_arrays(base_dir, "s", "symbols")
    return OnHWSymbolsDataset(
        X_train=Xtr,
        X_val=Xva,
        y_train=ytr,
        y_val=yva,
        train_ids=itr,
        val_ids=iva,
        split=split,
    )


def load_onhw_equations(base_dir: str) -> OnHWEquationsDataset:
    """Load the OnHW-equations dataset (15-symbol charset).

    Same layouts and the same use of the official split as
    ``load_onhw_symbols``; the difference is the ``_e`` file suffix.

    Note that the ``_e`` pickles hold per-symbol slices of the equations,
    one label per sample, not whole equation strings. Reassembling equations
    needs the ``all_indices_e.txt`` mapping that ships alongside them, which
    this loader does not read - so treat the result as symbol
    classification, not sequence-to-sequence, until that mapping is wired up.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"directory not found: {base_dir}")
    Xtr, ytr, itr, Xva, yva, iva, split = _load_symbol_arrays(
        base_dir, "e", "equations"
    )

    def to_seq(arr):
        return [[int(v)] for v in arr]

    Y_train, Y_val = to_seq(ytr), to_seq(yva)
    return OnHWEquationsDataset(
        X_train=Xtr,
        X_val=Xva,
        Y_train=Y_train,
        Y_val=Y_val,
        train_words=[decode_symbol_seq(s) for s in Y_train],
        val_words=[decode_symbol_seq(s) for s in Y_val],
        train_ids=itr,
        val_ids=iva,
        split=split,
    )


# --------------------------------------------------------------------------- #
# Transfer learning helper (chars -> symbols)
# --------------------------------------------------------------------------- #
def build_transfer_model(
    pretrained_chars_model, n_classes: int = 15, freeze_epochs: int = 3
):
    """Build a transfer-learning model from a pretrained OnHW-chars model.

    Takes a trained ``cnn_bilstm`` model (from ``onhw_models.build_cnn_bilstm``)
    and produces a new model for symbol classification by:

    1. Copying all layers except the final Dense softmax head.
    2. Attaching a new Dense(n_classes, softmax) head.
    3. Freezing the conv+recurrent trunk for the first ``freeze_epochs``
       epochs (only the new head trains), then unfreezing for fine-tuning
       at a low learning rate.

    The returned object is the new Keras model with the trunk layers frozen;
    the caller is responsible for training it for ``freeze_epochs`` epochs,
    then unfreezing and continuing training.

    Parameters
    ----------
    pretrained_chars_model : keras.Model
        A model trained on OnHW-chars (52 classes). The conv+BiLSTM trunk
        will be reused; only the final classification head is replaced.
    n_classes : int, default 15
        Number of classes in the target task (15 for OnHW-symbols).
    freeze_epochs : int, default 3
        Number of epochs to train only the new head before unfreezing the
        trunk. This is informational only - the caller must unfreeze manually.

    Returns
    -------
    keras.Model
        New model with trunk frozen and a fresh ``n_classes``-way head.
    """
    # Local imports to avoid requiring TensorFlow at module load time.
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    if len(pretrained_chars_model.layers) < 3:
        raise ValueError(
            "expected a model ending in <trunk> -> ... -> Dense(softmax); got "
            f"{len(pretrained_chars_model.layers)} layers"
        )

    # Everything up to (but not including) the old classification head. For
    # cnn_bilstm that is Input -> Conv1D -> ... -> BiLSTM -> Dense -> Dropout.
    trunk = Model(
        pretrained_chars_model.input, pretrained_chars_model.layers[-2].output
    )

    # Clone before reusing. A functional Model built directly on another
    # model's tensors shares its *layer objects*, so freezing the trunk here
    # would freeze the pretrained model too, and fine-tuning this model would
    # overwrite the pretrained weights in place - leaving the caller with a
    # chars model that is silently no longer the one they trained. Cloning
    # gives fresh layers; the learned weights are then copied across.
    trunk_copy = tf.keras.models.clone_model(trunk)
    trunk_copy.set_weights(trunk.get_weights())

    new_output = layers.Dense(n_classes, activation="softmax", name="transfer_softmax")(
        trunk_copy.output
    )
    new_model = Model(trunk_copy.input, new_output, name="transfer_from_chars")

    # Freeze the trunk; only the new head trains during the warmup epochs.
    for layer in new_model.layers[:-1]:
        layer.trainable = False
    new_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    print(
        f"Transfer model: {new_model.count_params():,} params, "
        f"trunk frozen for first {freeze_epochs} epochs, then unfreeze "
        f"and fine-tune at lr=1e-4."
    )
    return new_model


def unfreeze_trunk(model, lr: float = 1e-4):
    """Unfreeze the trunk of a transfer-learning model for fine-tuning.

    Call this after ``freeze_epochs`` warmup epochs of training only the
    new head. The model is recompiled with a low learning rate so the
    pretrained features don't get washed out.
    """
    import tensorflow as tf

    for layer in model.layers[:-1]:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    """CLI: load an OnHW-symbols or equations folder and print a summary."""
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base_dir", help="path to extracted OnHW-symbols_equations folder")

    ap.add_argument(
        "--task",
        choices=["symbols", "equations"],
        default="symbols",
        help="which sub-dataset to load (default: symbols)",
    )
    args = ap.parse_args()

    if args.task == "symbols":
        ds = load_onhw_symbols(args.base_dir)
    else:
        ds = load_onhw_equations(args.base_dir)
    print(ds.summary())
    if args.task == "symbols":
        print(f"Classes ({ds.n_classes}): {SYMBOLS_VOCAB}")
        print(
            f"Train label balance: {np.bincount(ds.y_train, minlength=ds.n_classes).tolist()}"
        )
    else:
        print(f"First 5 train equations: {ds.train_words[:5]}")
        print(f"Lexicon size: {len(ds.lexicon)} unique equations")


if __name__ == "__main__":
    main()
