"""OnHW sequence-to-sequence recognition (words / equations / split-words) with CTC.

While ``onhw_models.py`` classifies a whole recording into ONE character class,
the sequence OnHW datasets (OnHW-words500, OnHW-wordsRandom, OnHW-equations,
OnHW-wordsTraj) label a recording with a STRING (a word or an equation). That
is a sequence-to-sequence problem: the model must emit a variable-length symbol
sequence from a variable-length IMU stream without any per-symbol alignment.

The standard approach — used by the OnHW benchmark papers (Ott et al., IJDAR
2022) and by REWI (Li et al., iWOAR 2025) — is a convolutional-recurrent
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

    python onhw_seq2seq.py --imu-file data/words_x.pkl --labels-file data/words_gt.pkl

No sequence dataset is bundled with this repo (download the OnHW words /
equations datasets from the Fraunhofer IIS OnHW page). To verify the pipeline
end-to-end without a download, run the built-in synthetic demo:

    python onhw_seq2seq.py --demo
"""
from __future__ import annotations

import argparse
import pickle
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import pad_sequences

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
            cur.append(min(prev[j] + 1,            # deletion
                           cur[j - 1] + 1,         # insertion
                           prev[j - 1] + (ca != cb)))  # substitution
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
def build_ctc_models(maxlen: int, n_symbols: int, rnn_units: int = 64,
                     rnn_layers: int = 2) -> Tuple[Model, Model, int]:
    """Build the CNN+BiLSTM CTC network.

    Returns (train_model, inference_model, downsampled_len). The inference
    model maps IMU input to per-frame softmax posteriors; the train model wraps
    it with the CTC loss (Keras-2 pattern: loss computed in a Lambda layer).
    """
    inp = layers.Input(shape=(maxlen, N_CHANNELS), name="imu")
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    for _ in range(rnn_layers):
        x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    # +1 output for the CTC blank symbol (Keras CTC puts blank at the LAST index)
    y_pred = layers.Dense(n_symbols + 1, activation="softmax", name="posteriors")(x)
    infer_model = Model(inp, y_pred, name="ctc_cnn_bilstm")

    down_len = maxlen // 4  # two MaxPooling1D(2) stages

    labels = layers.Input(name="labels", shape=(None,), dtype="int32")
    input_len = layers.Input(name="input_len", shape=(1,), dtype="int32")
    label_len = layers.Input(name="label_len", shape=(1,), dtype="int32")

    def ctc_lambda(args):
        yp, lab, il, ll = args
        return K.ctc_batch_cost(lab, yp, il, ll)

    loss_out = layers.Lambda(ctc_lambda, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_len, label_len])
    train_model = Model([inp, labels, input_len, label_len], loss_out)
    train_model.compile(optimizer="adam",
                        loss={"ctc": lambda y_true, y_out: y_out})
    return train_model, infer_model, down_len


def ctc_greedy_decode(infer_model: Model, X: np.ndarray, down_len: int,
                      charset: Charset, batch: int = 64) -> List[str]:
    """Greedy CTC decoding (collapse repeats, drop blanks) -> strings."""
    out: List[str] = []
    for i in range(0, len(X), batch):
        chunk = X[i:i + batch]
        preds = infer_model.predict(chunk, verbose=0)
        decoded, _ = K.ctc_decode(preds, input_length=np.full(len(chunk), down_len),
                                  greedy=True)
        seqs = K.get_value(decoded[0])
        out.extend(charset.decode([s for s in seq if s >= 0]) for seq in seqs)
    return out


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_sequences(imu_file: str, labels_file: str):
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
    motifs = {s: rng.normal(0, 1, size=(18, N_CHANNELS)).astype(np.float32)
              for s in symbols}
    x, labels = [], []
    for _ in range(n):
        length = rng.integers(3, 7)
        lab = "".join(rng.choice(symbols, size=length))
        seq = np.concatenate([motifs[s] + rng.normal(0, 0.3, motifs[s].shape)
                              for s in lab]).astype(np.float32)
        x.append(seq)
        labels.append(lab)
    return x, labels


def prepare(x, labels, charset: Charset, maxlen: int, train_idx):
    """Standardize per channel (train-fit only), pad IMU and label tensors."""
    scaler = StandardScaler()
    scaler.fit(np.vstack([x[i] for i in train_idx]))
    x_norm = [scaler.transform(s).astype(np.float32) for s in x]
    X = pad_sequences(x_norm, maxlen=maxlen, padding="post",
                      truncating="post", dtype="float32")
    encoded = [charset.encode(lab) for lab in labels]
    max_lab = max(len(e) for e in encoded)
    Y = pad_sequences(encoded, maxlen=max_lab, padding="post", value=0,
                      dtype="int32")
    label_len = np.array([[len(e)] for e in encoded], dtype=np.int32)
    return X, Y, label_len


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def run(x, labels, epochs: int, batch: int, maxlen: int, rnn_units: int,
        rnn_layers: int, seed: int) -> Tuple[float, float]:
    charset = Charset(labels)
    n = len(x)
    idx = np.arange(n)
    train, tmp = train_test_split(idx, test_size=0.4, random_state=seed)
    val, test = train_test_split(tmp, test_size=0.5, random_state=seed)

    maxlen = min(int(max(len(x[i]) for i in train)), maxlen)
    X, Y, label_len = prepare(x, labels, charset, maxlen, train)

    train_model, infer_model, down_len = build_ctc_models(
        maxlen, charset.size, rnn_units, rnn_layers)
    # CTC requires input_length >= label_length for every sample
    if down_len < label_len.max():
        raise ValueError(
            f"Downsampled length {down_len} < longest label {label_len.max()}; "
            f"raise --max-len")
    input_len = np.full((n, 1), down_len, dtype=np.int32)
    dummy = np.zeros((n, 1), dtype=np.float32)

    def feed(ids):
        return [X[ids], Y[ids], input_len[ids], label_len[ids]], dummy[ids]

    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    train_model.fit(*feed(train), validation_data=feed(val),
                    epochs=epochs, batch_size=batch, verbose=2, callbacks=[es])

    hyps = ctc_greedy_decode(infer_model, X[test], down_len, charset)
    refs = [labels[i] for i in test]
    c, w = cer(refs, hyps), wer(refs, hyps)
    print(f"\nTest CER: {c * 100:.2f}%   Test WER: {w * 100:.2f}%   "
          f"(n={len(test)}, charset={charset.size} symbols)")
    for r, h in list(zip(refs, hyps))[:10]:
        print(f"  ref: {r!r:20s} hyp: {h!r}")
    return c, w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imu-file", help="pickle: list of (T,13) float arrays")
    ap.add_argument("--labels-file", help="pickle: list of label strings")
    ap.add_argument("--demo", action="store_true",
                    help="run on synthetic data to verify the CTC pipeline")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=800,
                    help="cap padded IMU length (words are much longer than chars)")
    ap.add_argument("--rnn-units", type=int, default=64)
    ap.add_argument("--rnn-layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if args.demo:
        x, labels = make_demo_data(seed=args.seed)
    elif args.imu_file and args.labels_file:
        x, labels = load_sequences(args.imu_file, args.labels_file)
    else:
        ap.error("provide --imu-file and --labels-file, or use --demo")

    print(f"Samples: {len(x)} | charset: {len(set(ch for l in labels for ch in l))} "
          f"symbols | mean IMU len: {np.mean([len(s) for s in x]):.0f}")
    run(x, labels, args.epochs, args.batch, args.max_len,
        args.rnn_units, args.rnn_layers, args.seed)


if __name__ == "__main__":
    main()
