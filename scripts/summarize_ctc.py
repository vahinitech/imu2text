"""Audit paired CTC predictions and summarize a preselected benchmark study."""

import argparse
import json
from pathlib import Path

import numpy as np

from imu2text.seq2seq import cer, wer, edit_distance
from imu2text.words import load_onhw_words500
from scripts.refit_ctc import executable_hash, file_hash


def audit_source(report, path):
    """Retain run-time hashes and add a fingerprint that tolerates prose-only edits."""
    if "source_sha256" not in report:
        return
    executable = report.get("executable_source_sha256", {})
    library = report.get("library_source_sha256", {})
    for name, expected in report["source_sha256"].items():
        source = Path(name).read_text(encoding="utf-8")
        library_hash = executable_hash(
            source, exclude_main=name == "imu2text/seq2seq.py"
        )
        if name in executable:
            if (
                executable_hash(source) != executable[name]
                and library.get(name) != library_hash
            ):
                raise ValueError(f"benchmark computation changed: {name}")
        elif file_hash(name) != expected:
            raise ValueError(f"cannot audit a changed benchmark source: {name}")
        executable.setdefault(name, executable_hash(source))
        library[name] = library_hash
    report["executable_source_sha256"] = executable
    report["library_source_sha256"] = library
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def character_check(directory):
    """Check paired character artifacts without using historical scores."""
    output, predictions = {}, {}
    reference, alphabet = None, None
    for name in ["chars_parent_seed0", "chars_fixed_seed0"]:
        with np.load(directory / f"{name}.npz") as saved:
            accuracy = float(np.mean(saved["true"] == saved["pred"])) * 100
            np.testing.assert_allclose(accuracy, saved["test_acc"], atol=1e-12, rtol=0)
            if reference is not None:
                np.testing.assert_array_equal(reference, saved["true"])
                np.testing.assert_array_equal(alphabet, saved["classes"])
            reference, alphabet = saved["true"], saved["classes"]
            predictions[name] = saved["pred"]
            output[name] = {
                "accuracy_percent": accuracy,
                "test": len(reference),
                "classes": len(alphabet),
            }
    output["changed_predictions"] = int(
        np.sum(predictions["chars_parent_seed0"] != predictions["chars_fixed_seed0"])
    )
    return output


def metrics(refs, hyps):
    """Compute sequence metrics directly from saved predictions."""
    if len(refs) != len(hyps) or not len(refs):
        raise ValueError("nonempty paired references and predictions are required")
    return {
        "cer": cer(list(refs), list(hyps)),
        "wer": wer(list(refs), list(hyps)),
        "word_accuracy": float(np.mean(np.asarray(refs) == np.asarray(hyps))),
    }


def decoder_changes(refs, greedy, lexicon):
    """Count exact-word recoveries and character edits when changing the decoder."""
    greedy_correct = np.asarray(refs) == np.asarray(greedy)
    lexicon_correct = np.asarray(refs) == np.asarray(lexicon)
    return {
        "gained_exact_words": int(np.sum(~greedy_correct & lexicon_correct)),
        "lost_exact_words": int(np.sum(greedy_correct & ~lexicon_correct)),
        "greedy_empty_decodes": int(np.sum(np.asarray(greedy) == "")),
        "lexicon_empty_decodes": int(np.sum(np.asarray(lexicon) == "")),
        "new_empty_decodes": int(
            np.sum((np.asarray(greedy) != "") & (np.asarray(lexicon) == ""))
        ),
        "character_edit_change": int(
            sum(
                edit_distance(r, after) - edit_distance(r, before)
                for r, before, after in zip(refs, greedy, lexicon)
            )
        ),
    }


def writer_bootstrap(refs, before, after, writers, repeats=5000):
    """Paired CER reduction interval, resampling whole test writers together."""
    groups = np.unique(writers)
    errors = np.array(
        [
            edit_distance(r, a) - edit_distance(r, b)
            for r, a, b in zip(refs, before, after)
        ]
    )
    characters = np.array([len(r) for r in refs])
    numerators = np.array([errors[writers == group].sum() for group in groups])
    denominators = np.array([characters[writers == group].sum() for group in groups])
    rng = np.random.default_rng(0)
    sampled = rng.integers(0, len(groups), size=(repeats, len(groups)))
    deltas = 100 * numerators[sampled].sum(axis=1) / denominators[sampled].sum(axis=1)
    return {
        "test_writers": len(groups),
        "resamples": repeats,
        "seed": 0,
        "cer_reduction_pp_95_interval": np.percentile(deltas, [2.5, 97.5]).tolist(),
    }


def main():
    """Check metrics, repeats, and length strata on the identical held-out samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--results", type=Path, default=Path("results/ctc"))
    args = parser.parse_args()
    ds = load_onhw_words500(args.data, fold=0)
    refs = np.array(ds.val_words)
    runs = {}
    predictions = {}
    histories = {}
    decoder_tradeoffs = {}
    names = [
        "parent_seed0_run1",
        "parent_seed0_run2",
        "fixed_seed0_run1",
        "fixed_seed0_run2",
        "fixed_random_seed0_run1",
    ]
    if (args.results / "refit_seed0.json").exists():
        names.append("refit_seed0")
    for name in names:
        report = json.loads((args.results / f"{name}.json").read_text(encoding="utf-8"))
        audit_source(report, args.results / f"{name}.json")
        histories[name] = report["history"]
        with np.load(args.results / f"{name}.json.predictions.npz") as saved:
            np.testing.assert_array_equal(saved["refs"], refs)
            predictions[name] = saved["hyps"]
            measured = metrics(refs, saved["hyps"])
            for metric, value in measured.items():
                np.testing.assert_allclose(value, report[metric], atol=1e-12, rtol=0)
            if "lexicon_hyps" in saved and len(saved["lexicon_hyps"]):
                measured["lexicon"] = metrics(refs, saved["lexicon_hyps"])
                for metric, value in measured["lexicon"].items():
                    np.testing.assert_allclose(
                        value, report["lexicon"][metric], atol=1e-12, rtol=0
                    )
                decoder_tradeoffs[name] = decoder_changes(
                    refs, saved["hyps"], saved["lexicon_hyps"]
                )
        runs[name] = measured
    repeat_checks = {}
    for variant in ["parent", "fixed"]:
        first, second = f"{variant}_seed0_run1", f"{variant}_seed0_run2"
        repeat_checks[variant] = {
            "identical_histories": histories[first] == histories[second],
            "different_predictions": int(
                np.sum(predictions[first] != predictions[second])
            ),
            "cer_difference_pp": 100 * abs(runs[first]["cer"] - runs[second]["cer"]),
        }
    lengths = np.array([len(sample) for sample in ds.X_val])
    strata = {}
    for label, mask in [
        ("short_at_most_80", lengths <= 80),
        ("native_81_to_800", (lengths > 80) & (lengths <= 800)),
        ("long_over_800", lengths > 800),
    ]:
        strata[label] = {"recordings": int(mask.sum())}
        if mask.any():
            for name in ["parent_seed0_run1", "fixed_seed0_run1", "refit_seed0"]:
                if name in predictions:
                    strata[label][name] = metrics(refs[mask], predictions[name][mask])
    summary = {
        "runs": runs,
        "repeat_checks": repeat_checks,
        "input_length_strata": strata,
        "decoder_tradeoffs": decoder_tradeoffs,
        "paired_writer_bootstrap": writer_bootstrap(
            refs,
            predictions["parent_seed0_run1"],
            predictions["fixed_seed0_run1"],
            ds.val_ids,
        ),
        "same_split_paired_writer_bootstrap": writer_bootstrap(
            refs,
            predictions["parent_seed0_run1"],
            predictions["fixed_random_seed0_run1"],
            ds.val_ids,
        ),
    }
    if "refit_seed0" in predictions:
        summary["refit_paired_writer_bootstrap"] = writer_bootstrap(
            refs,
            predictions["parent_seed0_run1"],
            predictions["refit_seed0"],
            ds.val_ids,
        )
    if (args.results / "chars_fixed_seed0.npz").exists():
        summary["character_checkpoint_check"] = character_check(args.results)
    (args.results / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
