"""OnHW character-recognition benchmark models (honest evaluation).

This module implements the OnHW-chars classification baselines and the
state-of-the-art **CNN+BiLSTM** architecture, evaluated the way the published
papers do: on a held-out test set, never on the training data.

Models implemented
------------------
- ``cnn``         : convolutional baseline (local features only)
- ``lstm``        : recurrent baseline (temporal only)
- ``bilstm``      : bidirectional recurrent baseline
- ``cnn_bilstm``  : SOTA convolutional-recurrent network (Ott et al., IJDAR 2022)

Why this exists
---------------
``cnn_gnn.py`` reports ~99% accuracy, but it trains and evaluates on the *same*
array, so that number is train-set memorization (held-out accuracy is ~43%).
This script fixes the methodology:

1. A stratified **train / val / test** split (test data is never seen in fit).
2. **Leak-free normalization**: a per-channel StandardScaler fit on the TRAIN
   split only, then applied to val/test.
3. **Early stopping** on validation accuracy with best-weight restore.
4. A side-by-side accuracy table across all architectures.

Augmentation
------------
Stochastic transforms live in ``onhw_augment.py`` and include the legacy
jitter / scale / magnitude-warp / time-warp policy plus three new physically-
meaningful IMU transforms:

- ``random_rotation``  - small 3D rotation of each Acc/Gyro/Mag triad
                         (simulates pen-grip variation; preserves vector norms)
- ``channel_dropout``  - zero a channel for the whole sample (sensor dropout)
- ``random_crop``      - random sub-window of the stroke (start/end are noisy)

Use ``--augment N`` to append N augmented copies of every training sample.

Limitation
----------
The bundled ``data/all_gt.pkl`` has no explicit writer IDs. This script infers them
heuristically from label order (see ``infer_writer_ids``) to approximate a
writer-independent split; if your data is not recorded in strict alphabet cycles,
prefer using true writer metadata and a group split on writer ID instead.

Usage
-----
    python onhw_models.py                 # train+eval all models, print table
    python onhw_models.py --models cnn_bilstm
    python onhw_models.py --epochs 80 --seed 1
    python onhw_models.py --augment 4 --rnn-units 100 --rnn-layers 2  # best config
"""
from __future__ import annotations

import argparse
import pickle
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
try:
    from tabulate import tabulate
except ImportError:  # optional dependency
    tabulate = None

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical, pad_sequences

# Augmentation policy + per-writer normalization live in a dedicated module so
# they can be reused by the seq2seq pipeline and unit-tested independently.
from onhw_augment import AUG_POLICIES, AugmentationConfig, augment_training

IMU_FILE = "data/all_x_dat_imu.pkl"
GT_FILE = "data/all_gt.pkl"
N_CHANNELS = 13

#: Below this many samples a per-writer scaler's statistics are too noisy to
#: beat the global one, so that writer falls back to the global train scaler.
MIN_SAMPLES_PER_WRITER_SCALER = 5


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_raw(imu_file: str = IMU_FILE,
             gt_file: str = GT_FILE) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """Load the IMU sequences and integer-encode the character labels (0..C-1)."""
    with open(imu_file, "rb") as f:
        x = [np.asarray(s, dtype=np.float32) for s in pickle.load(f)]
    with open(gt_file, "rb") as f:
        chars = list(pickle.load(f))
    classes = sorted(set(chars))                       # e.g. ['A'..'Z','a'..'z']
    char_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([char_to_idx[c] for c in chars], dtype=np.int64)
    return x, y, classes


def infer_writer_ids(chars: List[str]) -> np.ndarray:
    """Reconstruct per-sample writer IDs from the recording order.

    The OnHW pen records one writer at a time, who writes the alphabet
    sequentially (A..Z a..z). The bundled labels therefore appear in repeating
    alphabet cycles, but no explicit writer column is stored. We recover writer
    boundaries by detecting where a character *repeats* - that repeat marks the
    start of the next writer's session. This is what enables a true
    writer-independent (WI) split: no writer's samples can land in both train
    and test.
    """
    writer = np.empty(len(chars), dtype=np.int64)
    wid, seen = 0, set()
    for i, c in enumerate(chars):
        if c in seen:                                  # alphabet cycled -> new writer
            wid += 1
            seen = set()
        writer[i] = wid
        seen.add(c)
    return writer


def load_official_split(base_dir: str, case: str, dependency: str, fold: int,
                        seed: int, val_frac: float = 0.15, min_len: int = 1):
    """Load one of the 30 official OnHW-chars splits from the .npy release.

    Returns ``(x, y, classes, (train_idx, val_idx, test_idx))``.

    The published splits give train and test only. Early stopping needs a
    third set, so a stratified ``val_frac`` slice is carved out of the
    *training* half; the official test half is never touched. That keeps the
    reported test number comparable to published ones, at the cost of
    training on slightly less data than a published run would.

    ``dependency="indep"`` is the writer-independent protocol the OnHW papers
    report. These archives ship no per-sample writer IDs, so the writer
    partition is whatever the archive baked in - it cannot be re-derived
    here, which is exactly why the official folds are used as-is.
    """
    from onhw_chars import load_onhw_chars

    ds = load_onhw_chars(base_dir, case=case, dependency=dependency, fold=fold)
    if ds.format != "npy":
        raise SystemExit(
            f"{base_dir} is not the .npy OnHW-chars release (found "
            f"'{ds.format}'). The official splits only ship in the .npy "
            "archive - download it with `python onhw_download.py onhw_chars`.")

    x = [np.asarray(s_, dtype=np.float32) for s_ in ds.X_all]
    y = np.asarray(ds.y_all, dtype=np.int64)
    n_train = len(ds.X_train)

    # A handful of recordings in the published archive have zero timesteps
    # (3 of 31,275 in both/indep/fold0). They carry no signal, cannot be
    # normalized, and crash the scaler, so they are dropped - loudly, and with
    # the train/test counts printed, because dropping test samples changes the
    # denominator of any accuracy computed from them.
    keep = np.flatnonzero(np.array([len(s_) for s_ in x]) >= min_len)
    if len(keep) < len(x):
        kept_train = int(np.sum(keep < n_train))
        n_test = len(x) - n_train
        print(f"  dropped {len(x) - len(keep)} sample(s) with fewer than "
              f"{min_len} timestep(s): {n_train - kept_train} of {n_train} "
              f"train, {n_test - (len(keep) - kept_train)} of {n_test} test")
        x = [x[i] for i in keep]
        y = y[keep]
        n_train = kept_train

    train_all = np.arange(n_train)
    test_idx = np.arange(n_train, len(x))

    # Stratifying needs at least one validation slot per class; below that
    # sklearn refuses outright, so fall back to an unstratified split rather
    # than failing on a small case or a rare-class subset.
    n_val = max(1, int(round(len(train_all) * val_frac)))
    stratify = y[train_all] if n_val >= len(set(y[train_all].tolist())) else None
    tr, va = train_test_split(train_all, test_size=n_val,
                              random_state=seed, stratify=stratify)
    return x, y, list(ds.classes), (tr, va, test_idx)


def make_split(n: int, y: np.ndarray, seed: int, mode: str = "random",
               writers: np.ndarray = None):
    """Train / val / test index split (60 / 20 / 20).

    mode="random": stratified by class (a writer may appear in train and test).
    mode="writer": writer-independent - whole writers are assigned to exactly one
                   of train/val/test, so the test writers are entirely unseen.
                   This is the protocol the OnHW papers report as "WI".
    """
    if mode == "random":
        idx = np.arange(n)
        train, tmp = train_test_split(idx, test_size=0.40, random_state=seed, stratify=y)
        val, test = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=y[tmp])
        return train, val, test

    if mode == "writer":
        if writers is None:
            raise ValueError("writers must be provided for mode='writer'")
        # -1 marks "writer not recorded" (see onhw_chars.WRITER_UNKNOWN). Those
        # samples would all collapse into one pseudo-writer and turn a
        # writer-independent split into a leaky one, so refuse them outright.
        if np.any(np.asarray(writers) < 0):
            raise ValueError(
                "mode='writer' needs a real writer ID for every sample, but "
                "some are -1 (unknown). The .npy OnHW-chars archives ship "
                "pre-made splits instead of writer IDs - use their "
                "dependency='indep' folds for writer-independent evaluation.")
        uniq = np.unique(writers)
        if len(uniq) < 3:
            raise ValueError("mode='writer' requires at least 3 writers for a train/val/test split")
        train_w, tmp_w = train_test_split(uniq, test_size=0.40, random_state=seed, shuffle=True)
        val_w, test_w = train_test_split(tmp_w, test_size=0.50, random_state=seed, shuffle=True)
        sel = lambda group: np.flatnonzero(np.isin(writers, group))
        return sel(train_w), sel(val_w), sel(test_w)

    raise ValueError(f"unknown split mode: {mode}")


def normalize_and_pad(x: List[np.ndarray], train_idx: np.ndarray, maxlen: int,
                      writers: np.ndarray = None, mode: str = "global"):
    """Per-channel standardize then post-pad/truncate to ``maxlen``.

    Three normalization modes, differing in what statistics each sample is
    scaled by and, crucially, in what they assume about test-time access:

    - ``"global"`` (default): one ``StandardScaler`` fit on the train split's
      timesteps, applied to every sample. Strictly leak-free and symmetric -
      train and test are transformed identically. This is what the OnHW
      papers use and what every number in the README was measured with.
    - ``"per_sample"``: each sample is standardized by its *own* timesteps.
      Also strictly leak-free (a sample's own inputs are always available at
      inference), also symmetric, and it removes per-recording sensor bias
      without needing writer IDs at all. The trade-off is that it discards
      absolute signal level, which carries some class information.
    - ``"per_writer"``: each writer is standardized by their own timesteps,
      test writers included. **This is transductive**: it needs several
      samples from a test writer in hand before any of them can be
      normalized, so it does not describe single-shot inference on a fresh
      writer. It uses no labels, so it is not label leakage, but a number
      measured under it is not comparable to a standard writer-independent
      number and must be reported as transductive. Writers with fewer than
      ``MIN_SAMPLES_PER_WRITER_SCALER`` samples fall back to the global
      train scaler, since their statistics would be too noisy to trust.

    An earlier version of ``per_writer`` normalized train writers by their own
    statistics but test writers by the global train scaler. That is
    asymmetric - the model would be trained on per-writer-centred data and
    evaluated on globally-centred data - which is a distribution mismatch
    rather than an adaptation, so it is not offered.

    Returns the full padded tensor; callers slice it with the index splits.
    """
    if mode not in ("global", "per_sample", "per_writer"):
        raise ValueError(f"unknown normalization mode: {mode!r}")
    empty = [i for i, s in enumerate(x) if len(s) == 0]
    if empty:
        raise ValueError(
            f"{len(empty)} sample(s) have zero timesteps (first at index "
            f"{empty[0]}); they cannot be standardized. Filter them out before "
            "normalizing - the published OnHW-chars archive contains a few.")

    if mode == "per_sample":
        x_norm = [((s - s.mean(axis=0, keepdims=True))
                   / (s.std(axis=0, keepdims=True) + 1e-6)).astype(np.float32)
                  for s in x]
    elif mode == "per_writer":
        if writers is None:
            raise ValueError("mode='per_writer' requires writer IDs")
        if np.any(np.asarray(writers) < 0):
            # -1 marks "writer not recorded". Every such sample would share one
            # scaler fit across train and test together - not per-writer
            # adaptation at all, just a scaler that has seen the test set.
            raise ValueError(
                "mode='per_writer' needs a real writer ID for every sample, "
                "but some are -1 (unknown). The .npy OnHW-chars archives ship "
                "no writer IDs; use --norm global or per_sample with them.")
        global_scaler = StandardScaler()
        global_scaler.fit(np.vstack([x[i] for i in train_idx]))
        # Fit one scaler per writer from that writer's own timesteps, across
        # every split. No labels are involved, only inputs.
        per_w_scaler: Dict[int, StandardScaler] = {}
        for w in np.unique(writers):
            idxs = np.flatnonzero(writers == w)
            if len(idxs) < MIN_SAMPLES_PER_WRITER_SCALER:
                continue                     # too few samples -> use global
            sc = StandardScaler()
            sc.fit(np.vstack([x[i] for i in idxs]))
            per_w_scaler[int(w)] = sc
        x_norm = [per_w_scaler.get(int(writers[i]), global_scaler)
                  .transform(s).astype(np.float32)
                  for i, s in enumerate(x)]
    else:
        scaler = StandardScaler()
        scaler.fit(np.vstack([x[i] for i in train_idx]))   # fit on train timesteps only
        x_norm = [scaler.transform(s).astype(np.float32) for s in x]

    return pad_sequences(x_norm, maxlen=maxlen, padding="post",
                         truncating="post", dtype="float32")


# --------------------------------------------------------------------------- #
# Augmentation policy - re-exported from onhw_augment for backwards compat.
# --------------------------------------------------------------------------- #
# The legacy private transforms (_time_warp, _mag_warp, _augment_one) lived
# here in older revisions; they are now in onhw_augment. Tests that imported
# them as ``M._time_warp`` etc. keep working via these aliases.
from onhw_augment import (  # noqa: E402,F401
    time_warp as _time_warp,
    mag_warp as _mag_warp,
    augment_one as _augment_one,
)


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #
def _cnn_trunk(x):
    """Shared 1D-conv feature extractor (local stroke dynamics)."""
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    return x


def build_cnn(maxlen: int, n_classes: int) -> Model:
    inp = layers.Input(shape=(maxlen, N_CHANNELS))
    x = _cnn_trunk(inp)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inp, out, name="cnn")


# Recurrent capacity (CPU-friendly defaults; scale up on GPU via --rnn-units/--rnn-layers).
# The OnHW papers use ~2 layers of ~100 units; raise these to reproduce their capacity.
RNN_UNITS = 64
RNN_LAYERS = 1


def _stack_rnn(x, bidirectional: bool):
    """Stack RNN_LAYERS (Bi)LSTM layers; only the last returns a vector."""
    for i in range(RNN_LAYERS):
        last = i == RNN_LAYERS - 1
        cell = layers.LSTM(RNN_UNITS, return_sequences=not last)
        x = layers.Bidirectional(cell)(x) if bidirectional else cell(x)
    return x


def build_lstm(maxlen: int, n_classes: int) -> Model:
    inp = layers.Input(shape=(maxlen, N_CHANNELS))
    x = layers.Masking(mask_value=0.0)(inp)
    x = _stack_rnn(x, bidirectional=False)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inp, out, name="lstm")


def build_bilstm(maxlen: int, n_classes: int) -> Model:
    inp = layers.Input(shape=(maxlen, N_CHANNELS))
    x = layers.Masking(mask_value=0.0)(inp)
    x = _stack_rnn(x, bidirectional=True)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inp, out, name="bilstm")


def build_cnn_bilstm(maxlen: int, n_classes: int) -> Model:
    """SOTA: CNN trunk for local features -> stacked BiLSTM for temporal context."""
    inp = layers.Input(shape=(maxlen, N_CHANNELS))
    x = _cnn_trunk(inp)
    x = _stack_rnn(x, bidirectional=True)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inp, out, name="cnn_bilstm")


def build_cnn_bilstm_attn(maxlen: int, n_classes: int) -> Model:
    """CNN trunk -> BiLSTM over all timesteps -> attention-pooled classifier.

    ``cnn_bilstm`` reads out the BiLSTM's final state, which asks one vector
    to carry the whole stroke. A character's identity usually turns on a few
    moments of it - a crossbar, a loop, a pen lift - and where those fall
    varies with writing speed. Keeping the full output sequence and letting
    the model learn which timesteps matter suits that better than a fixed
    read-out position.

    Pooling is attention plus max, concatenated: attention gives a weighted
    average over timesteps, max picks up a sharp local event that averaging
    would dilute.
    """
    inp = layers.Input(shape=(maxlen, N_CHANNELS))
    x = _cnn_trunk(inp)
    for _ in range(RNN_LAYERS):
        x = layers.Bidirectional(
            layers.LSTM(RNN_UNITS, return_sequences=True))(x)

    score = layers.Dense(1, use_bias=False)(x)          # (B, T, 1)
    weights = layers.Softmax(axis=1)(score)             # over time
    context = layers.Dot(axes=1)([weights, x])          # (B, 1, 2*units)
    context = layers.Flatten()(context)
    pooled = layers.Concatenate()([context, layers.GlobalMaxPooling1D()(x)])

    h = layers.Dense(100, activation="relu")(pooled)
    h = layers.Dropout(0.3)(h)
    out = layers.Dense(n_classes, activation="softmax")(h)
    return Model(inp, out, name="cnn_bilstm_attn")


BUILDERS: Dict[str, Callable[[int, int], Model]] = {
    "cnn": build_cnn,
    "lstm": build_lstm,
    "bilstm": build_bilstm,
    "cnn_bilstm": build_cnn_bilstm,
    "cnn_bilstm_attn": build_cnn_bilstm_attn,
}


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def train_eval(name: str, X, Y, split, epochs: int, batch: int,
               label_smoothing: float = 0.0, lr_schedule: bool = False
               ) -> Dict[str, float]:
    """Train one model and report train/val/test accuracy.

    Two accuracy levers beyond the legacy defaults:

    - ``label_smoothing`` (default 0.0 = off): label smoothing regularizes the
      softmax by mixing the one-hot target with a uniform distribution. A
      value of 0.1 is the standard choice for classification with many
      classes. Unmeasured on this data - benchmark it before quoting a gain.
    - ``lr_schedule`` (default False): adds a ``ReduceLROnPlateau`` callback
      that halves the learning rate when validation accuracy plateaus. Helps
      the late-training regime where the default Adam LR is too aggressive.
    """
    tr, va, te = split
    print(f"  [{name}] training...", flush=True)
    maxlen, n_classes = X.shape[1], Y.shape[1]
    model = BUILDERS[name](maxlen, n_classes)
    loss = (tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
            if label_smoothing > 0 else "categorical_crossentropy")
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    callbacks = [EarlyStopping(monitor="val_accuracy", patience=8,
                               restore_best_weights=True, mode="max")]
    if lr_schedule:
        callbacks.append(ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                           patience=3, min_lr=1e-5, mode="max"))
    t0 = time.time()
    model.fit(X[tr], Y[tr], validation_data=(X[va], Y[va]),
              epochs=epochs, batch_size=batch, verbose=0, callbacks=callbacks)
    secs = time.time() - t0

    def acc(idx):
        pred = model.predict(X[idx], verbose=0)
        return float((np.argmax(pred, 1) == np.argmax(Y[idx], 1)).mean())

    result = {
        "model": name,
        "params": model.count_params(),
        "train_acc": acc(tr) * 100,
        "val_acc": acc(va) * 100,
        "test_acc": acc(te) * 100,
        "secs": secs,
        "test_pred": np.argmax(model.predict(X[te], verbose=0), 1),
        "test_true": np.argmax(Y[te], 1),
    }
    print(f"  [{name}] done: test={result['test_acc']:.2f}% "
          f"train={result['train_acc']:.2f}% ({secs:.0f}s)", flush=True)
    return result


def error_analysis(pred: np.ndarray, true: np.ndarray,
                   classes: List[str], top: int = 12) -> None:
    """Report where the errors go, not just how many there are.

    On the 52-class OnHW-chars task the class set is 26 letters in two cases,
    and case is largely a difference in *size*. Per-channel standardization
    removes absolute scale on purpose, so a case pair can be close to
    indistinguishable to the model. This prints how much of the error mass
    that accounts for: if case-insensitive accuracy is far above the plain
    figure, the headline number is limited by the label set rather than by
    the model, and more capacity or more epochs will not move it.
    """
    pred, true = np.asarray(pred), np.asarray(true)
    wrong = pred != true
    n_err = int(wrong.sum())
    print(f"\n  Error analysis on {len(true)} test samples "
          f"({n_err} wrong, {100 * n_err / len(true):.2f}% error)")
    if not n_err:
        return

    # Case-insensitive view: fold 'A' and 'a' onto one label and re-score.
    folded = {i: c.lower() for i, c in enumerate(classes)}
    ci_correct = sum(folded[int(p)] == folded[int(t)] for p, t in zip(pred, true))
    ci_acc = 100 * ci_correct / len(true)
    case_only = sum(1 for p, t in zip(pred[wrong], true[wrong])
                    if folded[int(p)] == folded[int(t)])
    print(f"  case-insensitive accuracy : {ci_acc:.2f}%  "
          f"(plain: {100 * (1 - n_err / len(true)):.2f}%)")
    print(f"  errors that are case only : {case_only}/{n_err} "
          f"({100 * case_only / n_err:.1f}% of all errors)")

    pairs: Dict[tuple, int] = {}
    for t, p in zip(true[wrong], pred[wrong]):
        pairs[(int(t), int(p))] = pairs.get((int(t), int(p)), 0) + 1
    ranked = sorted(pairs.items(), key=lambda kv: -kv[1])[:top]
    print(f"  top {len(ranked)} confusions (true -> predicted):")
    for (t, p), cnt in ranked:
        same = " [case pair]" if folded[t] == folded[p] else ""
        print(f"    {classes[t]!r} -> {classes[p]!r}: {cnt}"
              f" ({100 * cnt / n_err:.1f}% of errors){same}")


def main() -> None:
    global RNN_UNITS, RNN_LAYERS, N_CHANNELS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(BUILDERS),
                    choices=list(BUILDERS), help="which architectures to run")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["random", "writer"], default="writer",
                    help="'writer' = writer-independent (unseen writers in test); "
                         "'random' = stratified random (easier, leaks writer style)")
    ap.add_argument("--max-len", type=int, default=100,
                    help="cap padded sequence length (mean len ~46) to bound RNN cost")
    ap.add_argument("--rnn-units", type=int, default=RNN_UNITS,
                    help="(Bi)LSTM hidden units (paper ~100; lower is faster on CPU)")
    ap.add_argument("--rnn-layers", type=int, default=RNN_LAYERS,
                    help="number of stacked (Bi)LSTM layers (paper ~2)")
    ap.add_argument("--augment", type=int, default=0,
                    help="augmented copies per TRAIN sample (0 = off)")
    ap.add_argument("--aug-policy", choices=sorted(AUG_POLICIES), default="legacy",
                    help="'legacy' (default) = jitter/scale/mag-warp/time-warp, "
                         "the policy behind the measured 71.6%% WI result; "
                         "'extended' adds rotation/channel-dropout/crop, which "
                         "are unmeasured on this subset")
    ap.add_argument("--onhw-chars", default=None, metavar="DIR",
                    help="use the official OnHW-chars .npy splits in DIR "
                         "(e.g. data/onhw-chars_2021-06-30) instead of the "
                         "pickle files and this script's own split")
    ap.add_argument("--case", choices=("lower", "upper", "both"), default="both",
                    help="--onhw-chars: character set (both = 52 classes)")
    ap.add_argument("--dependency", choices=("dep", "indep"), default="indep",
                    help="--onhw-chars: 'indep' is the writer-independent "
                         "protocol the OnHW papers report")
    ap.add_argument("--fold", type=int, default=0,
                    help="--onhw-chars: official fold index 0-4")
    ap.add_argument("--imu-file", default=IMU_FILE,
                    help="pickle: list of (T, channels) float arrays")
    ap.add_argument("--gt-file", default=GT_FILE,
                    help="pickle: list of character labels")
    ap.add_argument("--writers-file", default=None,
                    help="pickle: list of writer codes (one per sample); "
                         "if omitted, writer IDs are inferred from label cycles")
    ap.add_argument("--channels", type=int, default=N_CHANNELS,
                    help="sensor channels per timestep (13 = OnHW pen, 16 = Vahini pen)")
    # ---- new accuracy levers ----
    ap.add_argument("--norm", choices=("global", "per_sample", "per_writer"),
                    default="global",
                    help="input normalization: 'global' (default, leak-free, "
                         "what the README numbers use); 'per_sample' (each "
                         "sample by its own stats, leak-free); 'per_writer' "
                         "(each writer by their own stats - TRANSDUCTIVE, "
                         "needs several samples per test writer, so report "
                         "any resulting number as transductive)")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="label smoothing factor (0 = off; 0.1 is the standard pick)")
    ap.add_argument("--lr-schedule", action="store_true",
                    help="reduce LR on validation plateau (factor 0.5, patience 3)")
    ap.add_argument("--error-analysis", action="store_true",
                    help="after training, break the test errors down by "
                         "confusion pair and report case-insensitive accuracy")
    ap.add_argument("--deterministic", action="store_true",
                    help="make a run bit-reproducible (op determinism + single "
                         "thread). --seed alone does not: CPU thread scheduling "
                         "still moves the result by several points. Slower, so "
                         "use it when comparing configurations")
    args = ap.parse_args()

    RNN_UNITS, RNN_LAYERS = args.rnn_units, args.rnn_layers
    N_CHANNELS = args.channels

    # set_random_seed covers python's `random`, numpy and TF in one call.
    # tf.random.set_seed alone does NOT reach the Keras layer initializers, so
    # two runs with the same --seed started from different initial weights.
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    if args.deterministic:
        # Seeding alone does not make a CPU run repeatable: parallel reductions
        # accumulate in whatever order threads finish, so two runs with the
        # same seed can land points apart. Pinning op determinism and running
        # single-threaded makes a run bit-reproducible, which is what a
        # before/after comparison needs. It is slower, so it is opt-in.
        tf.config.experimental.enable_op_determinism()
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    if args.onhw_chars:
        x, y, classes, (tr, va, te) = load_official_split(
            args.onhw_chars, args.case, args.dependency, args.fold, args.seed)
        n, n_classes = len(x), len(classes)
        # No per-sample writer IDs ship with these archives; the split already
        # encodes the writer partition. -1 keeps per-writer normalization and
        # make_split(mode="writer") from silently treating them as one writer.
        writers = np.full(n, -1, dtype=np.int64)
        split_desc = f"official {args.case}/{args.dependency}/fold{args.fold}"
    else:
        x, y, classes = load_raw(args.imu_file, args.gt_file)
        n, n_classes = len(x), len(classes)
        if args.writers_file:
            with open(args.writers_file, "rb") as f:
                codes = np.array([str(c) for c in pickle.load(f)])
            if len(codes) != n:
                raise SystemExit(f"{len(codes)} writer codes for {n} samples")
            _, writers = np.unique(codes, return_inverse=True)  # codes -> int IDs
        else:
            with open(args.gt_file, "rb") as f:
                writers = infer_writer_ids(list(pickle.load(f)))
        tr, va, te = make_split(n, y, args.seed, mode=args.split, writers=writers)
        split_desc = args.split
    if x and x[0].shape[1] != N_CHANNELS:
        raise SystemExit(f"data has {x[0].shape[1]} channels but --channels={N_CHANNELS}")
    if args.augment:
        aug_cfg = AUG_POLICIES[args.aug_policy]()
        x, y, writers, tr = augment_training(x, y, writers, tr, args.augment,
                                             args.seed, cfg=aug_cfg)
    split = (tr, va, te)
    maxlen = min(int(max(len(x[i]) for i in split[0])), args.max_len)  # TRAIN max, capped
    X = normalize_and_pad(x, split[0], maxlen, writers=writers, mode=args.norm)
    Y = to_categorical(y, num_classes=n_classes)

    extras = []
    if args.norm != "global":
        extras.append(f"norm={args.norm}"
                      + (" (TRANSDUCTIVE)" if args.norm == "per_writer" else ""))
    if args.label_smoothing > 0: extras.append(f"label smoothing={args.label_smoothing}")
    if args.lr_schedule:        extras.append("LR-on-plateau")
    extras_str = (" | " + ", ".join(extras)) if extras else ""

    aug_desc = f"x{args.augment}" + (f" ({args.aug_policy})" if args.augment else "")
    # The .npy archives ship no writer IDs (all -1); counting uniques there
    # would report "1 writer", which reads as a real number and is not one.
    known_w = writers[:n][writers[:n] >= 0]
    w_desc = str(len(np.unique(known_w))) if len(known_w) else "not recorded"
    print(f"Samples: {n} | classes: {n_classes} | writers: {w_desc} "
          f"| split: {split_desc} | aug: {aug_desc} | maxlen: {maxlen} "
          f"| tr/va/te: {len(split[0])}/{len(split[1])}/{len(split[2])}{extras_str}")
    print(f"Majority-class baseline (test): "
          f"{np.bincount(y[split[2]]).max() / len(split[2]) * 100:.2f}%\n")

    rows = [train_eval(m, X, Y, split, args.epochs, args.batch,
                       label_smoothing=args.label_smoothing,
                       lr_schedule=args.lr_schedule)
            for m in args.models]
    rows.sort(key=lambda r: r["test_acc"], reverse=True)

    table = [[r["model"], f"{r['params']:,}", f"{r['train_acc']:.2f}",
              f"{r['val_acc']:.2f}", f"{r['test_acc']:.2f}", f"{r['secs']:.0f}s"]
             for r in rows]
    headers = ["Model", "Params", "Train %", "Val %", "Test % (held-out)", "Time"]
    if tabulate is not None:
        print(tabulate(table, headers=headers, tablefmt="github"))
    else:
        print("\t".join(headers))
        for row in table:
            print("\t".join(map(str, row)))
    best = rows[0]
    label = "writer-independent" if args.split == "writer" else "random split"
    print(f"\nBest held-out: {best['model']} @ {best['test_acc']:.2f}% "
          f"({n_classes}-class, {label})")
    if args.error_analysis:
        error_analysis(best["test_pred"], best["test_true"], classes)


if __name__ == "__main__":
    main()
