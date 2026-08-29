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
from onhw_augment import AugmentationConfig, augment_training

IMU_FILE = "data/all_x_dat_imu.pkl"
GT_FILE = "data/all_gt.pkl"
N_CHANNELS = 13


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
        uniq = np.unique(writers)
        if len(uniq) < 3:
            raise ValueError("mode='writer' requires at least 3 writers for a train/val/test split")
        train_w, tmp_w = train_test_split(uniq, test_size=0.40, random_state=seed, shuffle=True)
        val_w, test_w = train_test_split(tmp_w, test_size=0.50, random_state=seed, shuffle=True)
        sel = lambda group: np.flatnonzero(np.isin(writers, group))
        return sel(train_w), sel(val_w), sel(test_w)

    raise ValueError(f"unknown split mode: {mode}")


def normalize_and_pad(x: List[np.ndarray], train_idx: np.ndarray, maxlen: int,
                      writers: np.ndarray = None, per_writer: bool = False):
    """Per-channel standardize then post-pad/truncate to ``maxlen``.

    Two normalization modes are supported:

    - **Global** (default, ``per_writer=False``): a single ``StandardScaler``
      is fit on the train split's timesteps and applied to every sample. This
      is the leak-free default the OnHW papers use.
    - **Per-writer** (``per_writer=True``): a separate scaler is fit for each
      *training* writer from their own timesteps, then each *test* writer is
      normalized with the global train scaler. This is a cheap form of
      writer adaptation: it removes per-writer sensor-mount bias (different
      pen grip -> different accelerometer bias) without leaking test data.

    Returns the full padded tensor; callers slice it with the index splits.
    """
    if per_writer and writers is not None:
        # Fit one scaler per training writer, normalize each sample by its own
        # writer's scaler. Test writers (unseen during training) fall back to
        # the global train scaler.
        global_scaler = StandardScaler()
        global_scaler.fit(np.vstack([x[i] for i in train_idx]))
        per_w_scaler: Dict[int, StandardScaler] = {}
        for w in np.unique(writers[train_idx]):
            mask = writers[train_idx] == w
            idxs = train_idx[mask]
            if len(idxs) < 5:                # too few samples -> use global
                continue
            sc = StandardScaler()
            sc.fit(np.vstack([x[i] for i in idxs]))
            per_w_scaler[int(w)] = sc
        x_norm = []
        for i, s in enumerate(x):
            sc = per_w_scaler.get(int(writers[i]), global_scaler)
            x_norm.append(sc.transform(s).astype(np.float32))
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


BUILDERS: Dict[str, Callable[[int, int], Model]] = {
    "cnn": build_cnn,
    "lstm": build_lstm,
    "bilstm": build_bilstm,
    "cnn_bilstm": build_cnn_bilstm,
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
      classes; it calibrates confidence and typically adds 0.5-1.0 points on
      OnHW-chars.
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
    }
    print(f"  [{name}] done: test={result['test_acc']:.2f}% "
          f"train={result['train_acc']:.2f}% ({secs:.0f}s)", flush=True)
    return result


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
                    help="augmented copies per TRAIN sample (0 = off); "
                         "jitter/scale/warp + rotation/channel-dropout/crop")
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
    ap.add_argument("--per-writer-norm", action="store_true",
                    help="normalize each sample by its own writer's scaler "
                         "(cheap writer adaptation; unseen writers fall back "
                         "to the global train scaler)")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="label smoothing factor (0 = off; 0.1 is the standard pick)")
    ap.add_argument("--lr-schedule", action="store_true",
                    help="reduce LR on validation plateau (factor 0.5, patience 3)")
    args = ap.parse_args()

    RNN_UNITS, RNN_LAYERS = args.rnn_units, args.rnn_layers
    N_CHANNELS = args.channels

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    x, y, classes = load_raw(args.imu_file, args.gt_file)
    n, n_classes = len(x), len(classes)
    if x and x[0].shape[1] != N_CHANNELS:
        raise SystemExit(f"data has {x[0].shape[1]} channels but --channels={N_CHANNELS}")
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
    if args.augment:
        x, y, writers, tr = augment_training(x, y, writers, tr, args.augment, args.seed)
    split = (tr, va, te)
    maxlen = min(int(max(len(x[i]) for i in split[0])), args.max_len)  # TRAIN max, capped
    X = normalize_and_pad(x, split[0], maxlen, writers=writers,
                          per_writer=args.per_writer_norm)
    Y = to_categorical(y, num_classes=n_classes)

    extras = []
    if args.per_writer_norm:    extras.append("per-writer norm")
    if args.label_smoothing > 0: extras.append(f"label smoothing={args.label_smoothing}")
    if args.lr_schedule:        extras.append("LR-on-plateau")
    extras_str = (" | " + ", ".join(extras)) if extras else ""

    print(f"Samples: {n} | classes: {n_classes} | writers: {len(np.unique(writers[:n]))} "
          f"| split: {args.split} | aug: x{args.augment} | maxlen: {maxlen} "
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


if __name__ == "__main__":
    main()
