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

Limitation
----------
The bundled ``data/all_gt.pkl`` has no writer IDs, so a true *writer-independent*
(WI) split is not possible on this subset; we use a stratified random split.
Published WI numbers are therefore an upper reference, not a like-for-like target.
To run the official WI protocol, swap ``make_split`` for a GroupShuffleSplit on
writer IDs once the full OnHW-chars dataset (with writer metadata) is available.

Usage
-----
    python onhw_models.py                 # train+eval all models, print table
    python onhw_models.py --models cnn_bilstm
    python onhw_models.py --epochs 80 --seed 1
"""
from __future__ import annotations

import argparse
import pickle
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

IMU_FILE = "data/all_x_dat_imu.pkl"
GT_FILE = "data/all_gt.pkl"
N_CHANNELS = 13


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_raw() -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """Load the IMU sequences and integer-encode the character labels (0..C-1)."""
    with open(IMU_FILE, "rb") as f:
        x = [np.asarray(s, dtype=np.float32) for s in pickle.load(f)]
    with open(GT_FILE, "rb") as f:
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
    boundaries by detecting where a character *repeats* — that repeat marks the
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
    mode="writer": writer-independent — whole writers are assigned to exactly one
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


def normalize_and_pad(x: List[np.ndarray], train_idx: np.ndarray, maxlen: int):
    """Per-channel standardize (scaler fit on TRAIN only) then post-pad/truncate.

    Returns the full padded tensor; callers slice it with the index splits.
    """
    scaler = StandardScaler()
    scaler.fit(np.vstack([x[i] for i in train_idx]))   # fit on train timesteps only
    x_norm = [scaler.transform(s).astype(np.float32) for s in x]
    return pad_sequences(x_norm, maxlen=maxlen, padding="post",
                         truncating="post", dtype="float32")


# --------------------------------------------------------------------------- #
# Data augmentation (IMU time series) — applied to TRAIN samples only
# --------------------------------------------------------------------------- #
def _time_warp(seq, rng, sigma=0.2, knots=4):
    """Locally speed up / slow down the trajectory (smooth, monotonic warp)."""
    t = seq.shape[0]
    base = np.linspace(0.0, 1.0, t)
    knot_x = np.linspace(0.0, 1.0, knots + 2)
    knot_y = knot_x + rng.normal(0.0, sigma / knots, size=knots + 2)
    knot_y[0], knot_y[-1] = 0.0, 1.0
    knot_y = np.maximum.accumulate(knot_y)             # keep time monotonic
    warped = np.interp(base, knot_x, knot_y)
    out = np.empty_like(seq)
    for c in range(seq.shape[1]):
        out[:, c] = np.interp(warped, base, seq[:, c])
    return out


def _mag_warp(seq, rng, sigma=0.15, knots=4):
    """Multiply the signal by a smooth random envelope (sensor gain drift)."""
    t = seq.shape[0]
    knot_x = np.linspace(0.0, 1.0, knots + 2)
    curve = np.interp(np.linspace(0.0, 1.0, t), knot_x,
                      rng.normal(1.0, sigma, size=knots + 2))
    return seq * curve[:, None]


def _augment_one(seq, rng):
    """One randomly-transformed copy of a raw (T, C) IMU sequence.

    Jitter and scale are proportional to each channel's own std so the same
    settings work across accelerometer / gyro / force channels of different
    magnitudes. Warps are applied probabilistically.
    """
    s = seq.astype(np.float32).copy()
    std = s.std(axis=0, keepdims=True) + 1e-6
    s = s * rng.normal(1.0, 0.08, size=(1, s.shape[1]))          # per-channel scale
    s = s + rng.normal(0.0, 0.05, size=s.shape) * std           # proportional jitter
    if rng.random() < 0.7:
        s = _mag_warp(s, rng)
    if rng.random() < 0.7:
        s = _time_warp(s, rng)
    return s


def augment_training(x, y, writers, train_idx, n_aug, seed):
    """Append n_aug augmented copies of each training sample to the dataset.

    Original samples keep their indices (augmented ones are appended), so the
    val/test index arrays remain valid and never see augmented data.
    """
    rng = np.random.default_rng(seed)
    x = list(x)
    new_y, new_w, new_idx = [], [], []
    for j in train_idx:
        for _ in range(n_aug):
            x.append(_augment_one(x[j], rng))
            new_idx.append(len(x) - 1)
            new_y.append(y[j])
            new_w.append(writers[j])
    y = np.concatenate([y, np.array(new_y, dtype=y.dtype)]) if new_y else y
    writers = np.concatenate([writers, np.array(new_w, dtype=writers.dtype)]) if new_w else writers
    train_idx = np.concatenate([train_idx, np.array(new_idx, dtype=train_idx.dtype)])
    return x, y, writers, train_idx


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
def train_eval(name: str, X, Y, split, epochs: int, batch: int) -> Dict[str, float]:
    tr, va, te = split
    print(f"  [{name}] training...", flush=True)
    maxlen, n_classes = X.shape[1], Y.shape[1]
    model = BUILDERS[name](maxlen, n_classes)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    es = EarlyStopping(monitor="val_accuracy", patience=8,
                       restore_best_weights=True, mode="max")
    t0 = time.time()
    model.fit(X[tr], Y[tr], validation_data=(X[va], Y[va]),
              epochs=epochs, batch_size=batch, verbose=0, callbacks=[es])
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
    global RNN_UNITS, RNN_LAYERS
    ap = argparse.ArgumentParser(description=__doc__)
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
                    help="augmented copies per TRAIN sample (0 = off); jitter/scale/warp")
    args = ap.parse_args()

    RNN_UNITS, RNN_LAYERS = args.rnn_units, args.rnn_layers

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    x, y, classes = load_raw()
    n, n_classes = len(x), len(classes)
    with open(GT_FILE, "rb") as f:
        writers = infer_writer_ids(list(pickle.load(f)))
    tr, va, te = make_split(n, y, args.seed, mode=args.split, writers=writers)
    if args.augment:
        x, y, writers, tr = augment_training(x, y, writers, tr, args.augment, args.seed)
    split = (tr, va, te)
    maxlen = min(int(max(len(x[i]) for i in split[0])), args.max_len)  # TRAIN max, capped
    X = normalize_and_pad(x, split[0], maxlen)
    Y = to_categorical(y, num_classes=n_classes)

    print(f"Samples: {n} | classes: {n_classes} | writers: {len(np.unique(writers[:n]))} "
          f"| split: {args.split} | aug: x{args.augment} | maxlen: {maxlen} "
          f"| tr/va/te: {len(split[0])}/{len(split[1])}/{len(split[2])}")
    print(f"Majority-class baseline (test): "
          f"{np.bincount(y[split[2]]).max() / len(split[2]) * 100:.2f}%\n")

    rows = [train_eval(m, X, Y, split, args.epochs, args.batch) for m in args.models]
    rows.sort(key=lambda r: r["test_acc"], reverse=True)

    table = [[r["model"], f"{r['params']:,}", f"{r['train_acc']:.2f}",
              f"{r['val_acc']:.2f}", f"{r['test_acc']:.2f}", f"{r['secs']:.0f}s"]
             for r in rows]
    print(tabulate(
        table,
        headers=["Model", "Params", "Train %", "Val %", "Test % (held-out)", "Time"],
        tablefmt="github"))
    best = rows[0]
    label = "writer-independent" if args.split == "writer" else "random split"
    print(f"\nBest held-out: {best['model']} @ {best['test_acc']:.2f}% "
          f"(52-class, {label})")


if __name__ == "__main__":
    main()
