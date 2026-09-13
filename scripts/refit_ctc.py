"""Refit all official training writers after grouped validation selects the budget.

The selected epoch comes from a completed writer-validation run. Test metrics
are never used to choose the budget. Export weights and normalization statistics
alongside predictions so the final recognizer can be reused.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from imu2text import seq2seq as sequence
from imu2text.sequence_data import resample_bounds, validate_ctc_lengths
from imu2text.words import load_onhw_words500, LexiconDecoder


def selected_budget(report):
    """Use the validation-selected epoch, independent of every test score."""
    if report.get("validation_split") != "writer" or not report.get("deterministic"):
        raise ValueError(
            "refitting requires a deterministic writer-validation selection"
        )
    epoch = report["selected_epoch"]
    if not 1 <= epoch <= report["epochs_run"]:
        raise ValueError("selected epoch is outside the completed validation run")
    if epoch != int(np.argmin(report["history"]["val_loss"])) + 1:
        raise ValueError("selected epoch must minimize validation loss")
    return epoch


def file_hash(path):
    """Hash large archive files without loading them into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def executable_hash(source, exclude_main=False):
    """Fingerprint syntax; optionally omit the CLI that a library caller bypasses."""
    tree = ast.parse(source)
    if exclude_main:
        tree.body = [
            node
            for node in tree.body
            if not isinstance(node, ast.FunctionDef) or node.name != "main"
        ]
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def verify_selection(selection_path, data):
    """Require the same source and archive bytes used to select the epoch budget."""
    report = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_budget(report)
    hashes = report.get(
        "library_source_sha256",
        report.get("executable_source_sha256", report["source_sha256"]),
    )
    for path, expected in hashes.items():
        actual = (
            executable_hash(
                Path(path).read_text(encoding="utf-8"),
                exclude_main=(
                    "library_source_sha256" in report and path == "imu2text/seq2seq.py"
                ),
            )
            if "executable_source_sha256" in report
            else file_hash(path)
        )
        if actual != expected:
            raise ValueError(f"source changed since validation selection: {path}")
    protocol = json.loads(
        (selection_path.parent / "protocol.json").read_text(encoding="utf-8")
    )
    if protocol["fold"] != report["fold"]:
        raise ValueError("selection and archive protocol disagree on the fold")
    for name, expected in protocol["fold_file_sha256"].items():
        if file_hash(Path(data) / str(report["fold"]) / name) != expected:
            raise ValueError(f"archive changed since validation selection: {name}")
    return report


def score(refs, hyps):
    """Report edit rates and exact word matches separately."""
    return {
        "cer": sequence.cer(refs, hyps),
        "wer": sequence.wer(refs, hyps),
        "word_accuracy": float(np.mean(np.asarray(refs) == np.asarray(hyps))),
    }


def main():
    """Perform a final fit on the official training half, with a fixed epoch budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = verify_selection(args.selection, args.data)
    epochs = selected_budget(config)
    tf.keras.utils.set_random_seed(config["seed"])
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    ds = load_onhw_words500(args.data, fold=config["fold"])
    if not set(ds.train_ids).isdisjoint(ds.val_ids):
        raise ValueError("official test writers overlap training writers")
    if (ds.n_train, ds.n_val) != (config["archive_train"], config["test"]):
        raise ValueError("archive sizes differ from the validation run")
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    x = resample_bounds(
        list(ds.X_train) + list(ds.X_val), config["minlen"], config["maxlen"]
    )
    labels = ds.train_words + ds.val_words
    charset = sequence.Charset(["".join(config["charset"])])
    train = np.arange(ds.n_train)
    X, Y, label_lengths = sequence.prepare(x, labels, charset, config["maxlen"], train)
    lengths = np.array([[len(sample) // 4] for sample in x], dtype=np.int32)
    validate_ctc_lengths(labels, lengths)
    model, inference, _ = sequence.build_ctc_models(
        config["maxlen"], charset.size, config["rnn_units"], config["rnn_layers"]
    )
    history = model.fit(
        sequence.CTCBatches(
            [X, Y, lengths, label_lengths], train, config["batch"], seed=config["seed"]
        ),
        epochs=epochs,
        shuffle=False,
        verbose=2,
    )
    if not np.isfinite(history.history["loss"]).all():
        raise ValueError("non-finite training loss; no model exported")
    weights_path = str(args.output) + ".weights.h5"
    inference.save_weights(weights_path)
    scaler = StandardScaler()
    for index in train:
        scaler.partial_fit(x[index])
    np.savez_compressed(
        str(args.output) + ".normalization.npz", mean=scaler.mean_, scale=scaler.scale_
    )
    _, restored, _ = sequence.build_ctc_models(
        config["maxlen"], charset.size, config["rnn_units"], config["rnn_layers"]
    )
    restored.load_weights(weights_path)
    probe = [X[:2], lengths[:2]]
    np.testing.assert_allclose(
        inference.predict(probe, verbose=0),
        restored.predict(probe, verbose=0),
        atol=1e-6,
        rtol=1e-6,
    )
    train_hyps = sequence.ctc_greedy_decode(
        inference, X[: ds.n_train], lengths[: ds.n_train], charset
    )
    hyps = sequence.ctc_greedy_decode(
        inference, X[ds.n_train :], lengths[ds.n_train :], charset
    )
    decoder = LexiconDecoder(
        sorted(set(ds.train_words)), charset="".join(charset.symbols), beam_width=8
    )
    lexicon_hyps = decoder.decode(inference, X[ds.n_train :], lengths[ds.n_train :])
    report = {
        "implementation": "refit-all-training",
        "selection": args.selection.name,
        "selection_sha256": file_hash(args.selection),
        "epochs_run": epochs,
        "seed": config["seed"],
        "deterministic": True,
        "dataset": config["dataset"],
        "fold": config["fold"],
        "protocol": "official writer-independent",
        "configuration": {
            name: config[name]
            for name in [
                "minlen",
                "maxlen",
                "rnn_units",
                "rnn_layers",
                "batch",
                "charset",
            ]
        },
        "train": ds.n_train,
        "test": ds.n_val,
        "training_writers": len(set(ds.train_ids)),
        "test_writers": len(set(ds.val_ids)),
        "training_metrics": score(ds.train_words, train_hyps),
        **score(ds.val_words, hyps),
        "lexicon": score(ds.val_words, lexicon_hyps),
        "lexicon_size": len(set(ds.train_words)),
        "history": history.history,
        "seconds": time.monotonic() - started,
        "weights_sha256": file_hash(weights_path),
        "refit_script_sha256": file_hash(__file__),
        "weights_file": Path(weights_path).name,
        "normalization_file": args.output.name + ".normalization.npz",
        "weights_reload_verified": True,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        str(args.output) + ".predictions.npz",
        refs=ds.val_words,
        hyps=hyps,
        lexicon_hyps=lexicon_hyps,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
