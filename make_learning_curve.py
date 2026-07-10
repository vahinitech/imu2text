"""Writer-independent learning curve for the smart-pen accuracy projection.

Trains the SOTA CNN+BiLSTM on an increasing number of *training writers*, while
always evaluating on the SAME held-out (unseen) test writers. The resulting
(writers, samples, accuracy) curve is what the MATLAB projection extrapolates to
full-dataset scale.

Output: results/learning_curve.csv  with columns
    n_writers, n_samples, wi_test_acc

Run:  python make_learning_curve.py
"""
import csv
import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

import onhw_models as M

SEED = 0
EPOCHS = 40
BATCH = 64
MAXLEN = 100
FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]


def main() -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.makedirs("results", exist_ok=True)

    x, y, classes = M.load_raw()
    n, n_classes = len(x), len(classes)
    with open(M.GT_FILE, "rb") as f:
        writers = M.infer_writer_ids(list(pickle.load(f)))

    tr_full, va, te = M.make_split(n, y, SEED, mode="writer", writers=writers)
    train_writers = np.unique(writers[tr_full])
    rng = np.random.default_rng(SEED)
    rng.shuffle(train_writers)

    Y = to_categorical(y, num_classes=n_classes)
    maxlen = min(int(max(len(x[i]) for i in tr_full)), MAXLEN)
    rows = []
    for frac in FRACTIONS:
        k = max(1, round(frac * len(train_writers)))
        chosen = set(train_writers[:k])
        tr = np.array([i for i in tr_full if writers[i] in chosen])

        # normalization fit ONLY on this training subset -> no leakage
        X = M.normalize_and_pad(x, tr, maxlen)

        model = M.build_cnn_bilstm(maxlen, n_classes)
        model.compile(optimizer="adam", loss="categorical_crossentropy",
                      metrics=["accuracy"])
        es = EarlyStopping(monitor="val_accuracy", patience=8,
                           restore_best_weights=True, mode="max")
        model.fit(X[tr], Y[tr], validation_data=(X[va], Y[va]),
                  epochs=EPOCHS, batch_size=BATCH, verbose=0, callbacks=[es])
        pred = model.predict(X[te], verbose=0)
        acc = float((np.argmax(pred, 1) == np.argmax(Y[te], 1)).mean())
        rows.append((k, len(tr), acc * 100))
        print(f"  writers={k:2d}  samples={len(tr):4d}  WI_test_acc={acc*100:5.2f}%",
              flush=True)

    with open("results/learning_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_writers", "n_samples", "wi_test_acc"])
        w.writerows(rows)
    print("\nWrote results/learning_curve.csv")


if __name__ == "__main__":
    main()
