"""OnHW sequence-to-sequence recognition (words / equations / split-words) with CTC.

While ``imu2text/models.py`` classifies a whole recording into ONE character class,
the sequence OnHW datasets (OnHW-words500, OnHW-wordsRandom, OnHW-equations,
OnHW-wordsTraj) label a recording with a STRING (a word or an equation). That
is a sequence-to-sequence problem: the model must emit a variable-length symbol
sequence from a variable-length IMU stream without any per-symbol alignment.

The standard approach - used by the OnHW benchmark papers (Ott et al., IJDAR
2022) and by REWI (Li et al., iWOAR 2025) - is a convolutional-recurrent
encoder trained with **CTC** (Connectionist Temporal Classification):

    IMU (T, 13) -> CNN trunk (local stroke features, downsamples time)
               -> stacked BiLSTM (temporal context)
               -> per-frame softmax over |charset| + 1 (CTC blank)
               -> CTC loss / greedy or beam-search decoding

Metrics are Character Error Rate (CER) and Word Error Rate (WER), the standard
handwriting-recognition metrics, computed via edit distance.

Data format
-----------
Same convention as the character pipeline: two pickles, one with a list of
(T_i, 13) float arrays and one with a list of label STRINGS, e.g.

    python -m imu2text.seq2seq --imu-file data/words_x.pkl --labels-file data/words_gt.pkl

No sequence dataset is bundled with this repo (download the OnHW words /
equations datasets from the Fraunhofer IIS OnHW page). To verify the pipeline
end-to-end without a download, run the built-in synthetic demo:

    python -m imu2text.seq2seq --demo
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import pad_sequences, Sequence as KerasSequence

from .callbacks import RestoreBest
from .sequence_data import (
    decoder_lengths,
    resample_bounds,
    sequence_split,
    validate_ctc_lengths,
)

N_CHANNELS = 13


# --------------------------------------------------------------------------- #
# Charset and metrics
# --------------------------------------------------------------------------- #
class Charset:
    """Bidirectional symbol <-> integer mapping. CTC blank = index ``size``."""

    def __init__(self, labels: Sequence[str]):
        self.symbols: List[str] = sorted({ch for lab in labels for ch in lab})
        self._to_idx: Dict[str, int] = {s: i for i, s in enumerate(self.symbols)}

    @property
    def size(self) -> int:
        return len(self.symbols)

    def encode(self, label: str) -> List[int]:
        return [self._to_idx[ch] for ch in label]

    def decode(self, indices: Sequence[int]) -> str:
        return "".join(self.symbols[i] for i in indices if 0 <= i < self.size)


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance between two sequences (insert/delete/substitute)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(
                    prev[j] + 1,  # deletion
                    cur[j - 1] + 1,  # insertion
                    prev[j - 1] + (ca != cb),
                )
            )  # substitution
        prev = cur
    return prev[-1]


def cer(refs: List[str], hyps: List[str]) -> float:
    """Character error rate: total edit distance / total reference length."""
    dist = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    total = sum(len(r) for r in refs)
    return dist / max(total, 1)


def wer(refs: List[str], hyps: List[str]) -> float:
    """Word error rate (words split on whitespace; single words -> 0/1 match)."""
    dist = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    total = sum(len(r.split()) for r in refs)
    return dist / max(total, 1)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_ctc_models(
    maxlen: int, n_symbols: int, rnn_units: int = 64, rnn_layers: int = 2
) -> Tuple[Model, Model, int]:
    """Build the CNN+BiLSTM CTC network.

    Returns (train_model, inference_model, downsampled_len). The inference
    model takes [padded_imu, output_lengths] and returns per-frame posteriors.
    Lengths have shape (batch, 1), measured after the two pooling stages.
    The train model wraps it with CTC loss computed in a Lambda layer.
    """
    inp = layers.Input(shape=(None, N_CHANNELS), name="imu")
    input_len = layers.Input(name="input_len", shape=(1,), dtype="int32")
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    mask = layers.Lambda(
        lambda inputs: tf.sequence_mask(
            tf.squeeze(inputs[0], -1), tf.shape(inputs[1])[1]
        ),
        name="valid_frames",
    )([input_len, x])
    for _ in range(rnn_layers):
        x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(
            x, mask=mask
        )
    x = layers.Dropout(0.3)(x)
    # +1 output for the CTC blank symbol (Keras CTC puts blank at the LAST index)
    y_pred = layers.Dense(n_symbols + 1, activation="softmax", name="posteriors")(x)
    infer_model = Model([inp, input_len], y_pred, name="ctc_cnn_bilstm")

    down_len = maxlen // 4  # two MaxPooling1D(2) stages

    labels = layers.Input(name="labels", shape=(None,), dtype="int32")
    label_len = layers.Input(name="label_len", shape=(1,), dtype="int32")

    def ctc_lambda(args):
        yp, lab, il, ll = args
        return K.ctc_batch_cost(lab, yp, il, ll)

    loss_out = layers.Lambda(ctc_lambda, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_len, label_len]
    )
    train_model = Model([inp, labels, input_len, label_len], loss_out)
    train_model.compile(optimizer="adam", loss={"ctc": lambda y_true, y_out: y_out})
    return train_model, infer_model, down_len


def ctc_greedy_decode(
    infer_model: Model, X: np.ndarray, down_len: int, charset: Charset, batch: int = 64
) -> List[str]:
    """Decode up to each recording's output length, excluding padded frames."""
    out: List[str] = []
    lengths = decoder_lengths(down_len, len(X))
    for i in range(0, len(X), batch):
        chunk = X[i : i + batch]
        chunk_lengths = lengths[i : i + batch]
        chunk = chunk[:, : int(chunk_lengths.max()) * 4 + 3]
        preds = infer_model.predict([chunk, chunk_lengths[:, None]], verbose=0)
        if np.any(chunk_lengths > preds.shape[1]):
            raise ValueError("decode length exceeds model output")
        decoded, _ = K.ctc_decode(preds, input_length=chunk_lengths, greedy=True)
        seqs = K.get_value(decoded[0])
        out.extend(charset.decode([s for s in seq if s >= 0]) for seq in seqs)
    return out


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_sequences(imu_file: str, labels_file: str):
    """Load IMU sequences and their transcriptions from a pair of pickles."""
    with open(imu_file, "rb") as f:
        x = [np.asarray(s, dtype=np.float32) for s in pickle.load(f)]
    with open(labels_file, "rb") as f:
        labels = [str(s) for s in pickle.load(f)]
    if len(x) != len(labels):
        raise ValueError(f"{len(x)} IMU samples but {len(labels)} labels")
    return x, labels


def make_demo_data(n: int = 240, seed: int = 0):
    """Synthetic 'equations': each symbol has a fixed random IMU motif, samples
    are motif concatenations + noise. Lets the CTC pipeline be verified without
    downloading the real OnHW sequence datasets."""
    rng = np.random.default_rng(seed)
    symbols = list("0123456789+-=")
    motifs = {
        s: rng.normal(0, 1, size=(18, N_CHANNELS)).astype(np.float32) for s in symbols
    }
    x, labels = [], []
    for _ in range(n):
        length = rng.integers(3, 7)
        lab = "".join(rng.choice(symbols, size=length))
        seq = np.concatenate(
            [motifs[s] + rng.normal(0, 0.3, motifs[s].shape) for s in lab]
        ).astype(np.float32)
        x.append(seq)
        labels.append(lab)
    return x, labels


def prepare(x, labels, charset: Charset, maxlen: int, train_idx):
    """Standardize per channel (train-fit only), pad IMU and label tensors."""
    scaler = StandardScaler()
    # Incremental fitting avoids a second copy of the full training archive.
    for i in train_idx:
        scaler.partial_fit(x[i])
    X = np.zeros((len(x), maxlen, N_CHANNELS), dtype=np.float32)
    for i, sample in enumerate(x):
        size = min(len(sample), maxlen)
        X[i, :size] = scaler.transform(sample[:size])
    encoded = [charset.encode(lab) for lab in labels]
    max_lab = max(len(e) for e in encoded)
    Y = pad_sequences(encoded, maxlen=max_lab, padding="post", value=0, dtype="int32")
    label_len = np.array([[len(e)] for e in encoded], dtype=np.int32)
    return X, Y, label_len


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
class CTCBatches(KerasSequence):
    """Copy one minibatch at a time instead of duplicating full padded partitions."""

    def __init__(self, arrays, indices, batch, seed=None):
        self.arrays = arrays
        self.indices = np.array(indices, copy=True)
        self.batch = batch
        self.rng = np.random.default_rng(seed) if seed is not None else None
        self.on_epoch_end()

    def __len__(self):
        return (len(self.indices) + self.batch - 1) // self.batch

    def __getitem__(self, index):
        ids = self.indices[index * self.batch : (index + 1) * self.batch]
        inputs = [array[ids] for array in self.arrays]
        inputs[0] = inputs[0][:, : int(inputs[2].max()) * 4 + 3]
        return inputs, np.zeros((len(ids), 1), np.float32)

    def on_epoch_end(self):
        if self.rng is not None:
            self.rng.shuffle(self.indices)


def run(
    x,
    labels,
    epochs: int,
    batch: int,
    maxlen: int,
    rnn_units: int,
    rnn_layers: int,
    seed: int,
    n_train: int = None,
    lexicon: bool = False,
    *,
    writers=None,
    minlen: int = 80,
    results_path=None,
    symbols=None,
) -> Tuple[float, float]:
    """Train the CTC model on one dataset and return its test (CER, WER).

    ``n_train`` marks a pre-defined split: the first ``n_train`` samples are
    the shipped training half and everything after is the shipped test half.
    Pass it for archives whose split is writer-disjoint, otherwise a random
    re-split here would put the same writer on both sides and the number
    would no longer be writer-independent.

    ``writers`` enables writer-disjoint validation within the training half.
    ``symbols`` can supply a published alphabet; otherwise it is fitted on train.
    Duration bounds are fixed before examining labels, including held-out labels.
    """
    started = time.monotonic()
    n = len(x)
    if len(labels) != n:
        raise ValueError("one target string is required for each recording")
    if results_path:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    train, val, test = sequence_split(n, seed, n_train, writers)
    charset = Charset([symbols] if symbols is not None else [labels[i] for i in train])
    unknown = set("".join(labels)) - set(charset.symbols)
    if unknown:
        raise ValueError(
            f"held-out labels contain {len(unknown)} symbols absent from train"
        )
    original_lengths = np.array([len(sample) for sample in x])
    x = resample_bounds(x, minlen, maxlen)
    X, Y, label_len = prepare(x, labels, charset, maxlen, train)
    input_len = np.array([[len(sample) // 4] for sample in x], dtype=np.int32)
    validate_ctc_lengths(labels, input_len)

    train_model, infer_model, _ = build_ctc_models(
        maxlen, charset.size, rnn_units, rnn_layers
    )
    arrays = [X, Y, input_len, label_len]
    es = RestoreBest(monitor="val_loss", patience=8, restore_best_weights=True)
    history = train_model.fit(
        CTCBatches(arrays, train, batch, seed=seed),
        validation_data=CTCBatches(arrays, val, batch),
        epochs=epochs,
        shuffle=False,
        verbose=2,
        callbacks=[es],
    )

    hyps = ctc_greedy_decode(infer_model, X[test], input_len[test], charset)
    refs = [labels[i] for i in test]
    c, w = cer(refs, hyps), wer(refs, hyps)

    lex_metrics = None
    lex_hyps = []
    if lexicon:
        # Closed vocabulary: constrain the decode to words that exist. The
        # lexicon is built from the training half only - taking it from the
        # test labels too would leak which words are about to be scored.
        from .words import LexiconDecoder  # noqa: PLC0415

        decoder = LexiconDecoder(
            sorted(set(labels[i] for i in train)),
            charset="".join(charset.symbols),
            beam_width=8,
        )
        lex_hyps = decoder.decode(infer_model, X[test], input_len[test])
        lc, lw = cer(refs, lex_hyps), wer(refs, lex_hyps)
        lex_metrics = {
            "cer": lc,
            "wer": lw,
            "word_accuracy": float(np.mean(np.array(refs) == np.array(lex_hyps))),
        }
        print(
            f"\nLexicon-constrained: CER {lc * 100:.2f}%  WER {lw * 100:.2f}%  "
            f"(vocabulary of {len(set(labels[i] for i in train))} words from train)"
        )
    print(
        f"\nTest CER: {c * 100:.2f}%   Test WER: {w * 100:.2f}%   "
        f"(n={len(test)}, charset={charset.size} symbols)"
    )
    for r, h in list(zip(refs, hyps))[:10]:
        print(f"  ref: {r!r:20s} hyp: {h!r}")
    if results_path:
        report = {
            "cer": c,
            "wer": w,
            "word_accuracy": float(np.mean(np.array(refs) == np.array(hyps))),
            "lexicon": lex_metrics,
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "charset": charset.symbols,
            "seed": seed,
            "validation_split": "writer" if writers is not None else "random",
            "official_partition": n_train is not None,
            "epochs_requested": epochs,
            "epochs_run": len(history.history["loss"]),
            "selected_epoch": int(np.argmin(history.history["val_loss"])) + 1,
            "history": history.history,
            "seconds": time.monotonic() - started,
            "rnn_units": rnn_units,
            "rnn_layers": rnn_layers,
            "batch": batch,
            "minlen": minlen,
            "maxlen": maxlen,
            "resampled_short": int((original_lengths < minlen).sum()),
            "resampled_long": int((original_lengths > maxlen).sum()),
        }
        Path(results_path).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            str(results_path) + ".predictions.npz",
            refs=refs,
            hyps=hyps,
            lexicon_hyps=lex_hyps,
            train_indices=train,
            validation_indices=val,
            test_indices=test,
        )
    return c, w


def main() -> None:
    """CLI: run the synthetic demo, or train the CTC model on real data."""
    global N_CHANNELS
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--onhw-words500",
        default=None,
        metavar="DIR",
        help="extracted OnHW-Words500 folder; uses the archive's own "
        "writer-disjoint train/val split for the given --fold",
    )
    ap.add_argument("--fold", type=int, default=0, help="--onhw-words500: fold 0-4")
    ap.add_argument(
        "--lexicon",
        action="store_true",
        help="also decode constrained to the training vocabulary and report "
        "both, for closed-vocabulary datasets like OnHW-words500",
    )
    ap.add_argument("--imu-file", help="pickle: list of (T,13) float arrays")
    ap.add_argument("--labels-file", help="pickle: list of label strings")
    ap.add_argument(
        "--demo",
        action="store_true",
        help="run on synthetic data to verify the CTC pipeline",
    )
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument(
        "--max-len",
        type=int,
        default=800,
        help="resample longer recordings to this length, retaining the whole signal",
    )
    ap.add_argument("--rnn-units", type=int, default=64)
    ap.add_argument("--rnn-layers", type=int, default=2)
    ap.add_argument(
        "--channels",
        type=int,
        default=N_CHANNELS,
        help="sensor channels per timestep (13 = OnHW pen)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--min-len",
        type=int,
        default=80,
        help="resample shorter recordings to this fixed length",
    )
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--results", help="write metrics JSON and prediction/split NPZ")
    args = ap.parse_args()
    N_CHANNELS = args.channels

    # set_random_seed covers python's `random`, numpy and TF in one call.
    # tf.random.set_seed alone does NOT reach the Keras layer initialisers, so
    # the demo started from different weights every run and could fail
    # outright (100% CER) on an unlucky init. Same fix as imu2text/models.py.
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    words_split = None
    writers = None
    symbols = None
    if args.demo:
        x, labels = make_demo_data(seed=args.seed)
    elif args.onhw_words500:
        from .words import load_onhw_words500, WORDS500_VOCAB  # noqa: PLC0415

        ds = load_onhw_words500(args.onhw_words500, fold=args.fold)
        # The archive's split is the evaluation; run() must not re-split it.
        x = list(ds.X_train) + list(ds.X_val)
        labels = list(ds.train_words) + list(ds.val_words)
        words_split = len(ds.X_train)
        writers = np.concatenate([ds.train_ids, ds.val_ids])
        symbols = WORDS500_VOCAB
        print(
            f"OnHW-Words500 fold {args.fold}: train={ds.n_train} val={ds.n_val} "
            f"writers={ds.n_writers} lexicon={len(ds.lexicon)}"
        )
    elif args.imu_file and args.labels_file:
        x, labels = load_sequences(args.imu_file, args.labels_file)
        if x and x[0].shape[1] != N_CHANNELS:
            ap.error(f"data has {x[0].shape[1]} channels but --channels={N_CHANNELS}")
    else:
        ap.error("provide --imu-file and --labels-file, or use --demo")

    print(
        f"Samples: {len(x)} | charset: {len(set(ch for l in labels for ch in l))} "
        f"symbols | mean IMU len: {np.mean([len(s) for s in x]):.0f}"
    )
    run(
        x,
        labels,
        args.epochs,
        args.batch,
        args.max_len,
        args.rnn_units,
        args.rnn_layers,
        args.seed,
        n_train=words_split,
        lexicon=args.lexicon,
        writers=writers,
        minlen=args.min_len,
        results_path=args.results,
        symbols=symbols,
    )


if __name__ == "__main__":
    main()
