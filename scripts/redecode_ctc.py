"""Decode a saved Words500 refit again, with the exact lexicon decoder.

``refit_ctc`` exports the inference weights (ignored by Git) and records their
checksum. This loads them, checks the checksum, rebuilds the test inputs the
same way, confirms that greedy decoding still matches the saved predictions,
and then decodes the test half with ``LexiconDecoder``'s exact word scoring.
Nothing is trained and nothing is fitted on the test half; the lexicon is the
training words.

    python -m scripts.redecode_ctc --data data/Words500_indep_02 \\
        --refit results/ctc/refit_seed0.json \\
        --weights results/ctc/refit_seed0.json.weights.h5 \\
        --output results/ctc/refit_seed0_exact.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from imu2text import seq2seq as sequence
from imu2text.sequence_data import resample_bounds
from imu2text.words import LexiconDecoder, load_onhw_words500
from scripts.refit_ctc import file_hash, score


def main():
    """Re-decode the refit's test half and write a report plus predictions."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--refit", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    refit = json.loads(args.refit.read_text(encoding="utf-8"))
    if file_hash(args.weights) != refit["weights_sha256"]:
        raise ValueError(f"{args.weights} is not the model {args.refit} recorded")
    saved = np.load(str(args.refit) + ".predictions.npz", allow_pickle=False)
    config = refit["configuration"]
    ds = load_onhw_words500(args.data, fold=refit["fold"])
    x = resample_bounds(
        list(ds.X_train) + list(ds.X_val), config["minlen"], config["maxlen"]
    )
    charset = sequence.Charset(["".join(config["charset"])])
    X, _, _ = sequence.prepare(
        x,
        ds.train_words + ds.val_words,
        charset,
        config["maxlen"],
        np.arange(ds.n_train),
    )
    lengths = np.array([[len(s) // 4] for s in x], dtype=np.int32)
    _, inference, _ = sequence.build_ctc_models(
        config["maxlen"], charset.size, config["rnn_units"], config["rnn_layers"]
    )
    inference.load_weights(str(args.weights))

    test_x, test_lengths = X[ds.n_train :], lengths[ds.n_train :]
    refs = ds.val_words
    if list(refs) != saved["refs"].tolist():
        raise ValueError("test words differ from the saved predictions")
    greedy = sequence.ctc_greedy_decode(inference, test_x, test_lengths, charset)
    if greedy != saved["hyps"].tolist():
        raise ValueError("greedy decoding no longer matches the saved predictions")
    lexicon = sorted(set(ds.train_words))
    started = time.monotonic()
    decoder = LexiconDecoder(lexicon, charset="".join(charset.symbols))
    exact = decoder.decode(inference, test_x, test_lengths)
    report = {
        "implementation": "exact-lexicon-redecode",
        "refit": args.refit.name,
        "refit_sha256": file_hash(args.refit),
        "weights_sha256": refit["weights_sha256"],
        "dataset": refit["dataset"],
        "fold": refit["fold"],
        "protocol": refit["protocol"],
        "seed": refit["seed"],
        "test": len(refs),
        "lexicon_size": len(lexicon),
        "lexicon_source": "training words only",
        "greedy_matches_saved": True,
        "metrics": {
            "greedy": score(refs, greedy),
            "beam8_strict_saved": score(refs, saved["lexicon_hyps"].tolist()),
            "exact_lexicon": score(refs, exact),
        },
        "empty_outputs": {
            "greedy": sum(1 for h in greedy if not h),
            "beam8_strict_saved": int(np.sum(saved["lexicon_hyps"] == "")),
            "exact_lexicon": sum(1 for h in exact if not h),
        },
        "decode_seconds": round(time.monotonic() - started, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        str(args.output) + ".predictions.npz",
        refs=np.asarray(refs),
        hyps=np.asarray(greedy),
        lexicon_hyps=np.asarray(exact),
    )
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
