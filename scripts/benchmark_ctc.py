"""Compare the parent and corrected CTC pipelines on an official Words500 fold.

The parent implementation is read from this repository's Git history. Its model,
preprocessing, split, loss, and decoding execute unchanged. Wrappers record the
training history and predictions and enable deterministic TensorFlow execution.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time
import types

import numpy as np
import tensorflow as tf

from imu2text import seq2seq
from imu2text.words import load_onhw_words500, WORDS500_VOCAB


def parent_module(revision):
    """Read this project's historical implementation without changing the checkout."""
    source = subprocess.run(
        ["git", "show", f"{revision}:imu2text/seq2seq.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = types.ModuleType("imu2text._benchmark_parent")
    module.__package__ = "imu2text"
    exec(compile(source, f"{revision}:imu2text/seq2seq.py", "exec"), module.__dict__)
    return module


def run_parent(args, ds):
    """Capture the parent's metrics without altering its training computation."""
    # The historical module's attributes are created by exec, not static imports.
    # pylint: disable=no-member
    module = parent_module(args.parent)
    history = {}
    predictions = []
    original_fit = module.Model.fit
    original_decode = module.ctc_greedy_decode

    def record_fit(self, *positional, **kwargs):
        result = original_fit(self, *positional, **kwargs)
        history.update(result.history)
        return result

    def record_decode(*positional, **kwargs):
        result = original_decode(*positional, **kwargs)
        predictions.extend(result)
        return result

    module.Model.fit = record_fit
    module.ctc_greedy_decode = record_decode
    try:
        c, w = module.run(
            list(ds.X_train) + list(ds.X_val),
            ds.train_words + ds.val_words,
            args.epochs,
            args.batch,
            args.max_len,
            args.rnn_units,
            args.rnn_layers,
            args.seed,
            n_train=ds.n_train,
        )
    finally:
        module.Model.fit = original_fit
    np.savez_compressed(
        str(args.output) + ".predictions.npz", refs=ds.val_words, hyps=predictions
    )
    return {
        "cer": c,
        "wer": w,
        "history": history,
        "word_accuracy": float(np.mean(np.array(ds.val_words) == predictions)),
        "validation_split": "random",
        "epochs_run": len(history["loss"]),
    }


def main():
    """Run one preselected configuration; do not select on the test result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--implementation", choices=["parent", "fixed", "fixed-random"], required=True
    )
    parser.add_argument("--parent", default="03d1c8f")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=800)
    parser.add_argument("--min-len", type=int, default=80)
    parser.add_argument("--rnn-units", type=int, default=32)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--lexicon", action="store_true")
    args = parser.parse_args()
    if args.lexicon and args.implementation == "parent":
        parser.error("parent runs record greedy metrics only")
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    ds = load_onhw_words500(args.data, fold=args.fold)
    if not set(ds.train_ids).isdisjoint(ds.val_ids):
        parser.error("benchmark requires an official writer-independent archive")
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.implementation == "parent":
        report = run_parent(args, ds)
    else:
        writers = (
            np.concatenate([ds.train_ids, ds.val_ids])
            if args.implementation == "fixed"
            else None
        )
        seq2seq.run(
            list(ds.X_train) + list(ds.X_val),
            ds.train_words + ds.val_words,
            args.epochs,
            args.batch,
            args.max_len,
            args.rnn_units,
            args.rnn_layers,
            args.seed,
            n_train=ds.n_train,
            lexicon=args.lexicon,
            writers=writers,
            minlen=args.min_len,
            results_path=args.output,
            symbols=WORDS500_VOCAB,
        )
        report = json.loads(args.output.read_text(encoding="utf-8"))
    report.update(
        {
            "implementation": args.implementation,
            "parent_commit": args.parent,
            "dataset": Path(args.data).name,
            "fold": args.fold,
            "protocol": "official writer-independent",
            "seed": args.seed,
            "deterministic": True,
            "epochs_requested": args.epochs,
            "rnn_units": args.rnn_units,
            "rnn_layers": args.rnn_layers,
            "batch": args.batch,
            "maxlen": args.max_len,
            "archive_train": ds.n_train,
            "test": ds.n_val,
            "writers": ds.n_writers,
            "symbols": len(WORDS500_VOCAB),
            "observed_lexicon_size": len(ds.lexicon),
            "seconds": time.monotonic() - started,
            "environment": {
                "python": platform.python_version(),
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (
                    Path("imu2text/seq2seq.py"),
                    Path("imu2text/callbacks.py"),
                    Path("imu2text/sequence_data.py"),
                    Path("imu2text/words.py"),
                    Path("scripts/benchmark_ctc.py"),
                )
            },
        }
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
