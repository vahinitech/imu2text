"""Average the saved softmax outputs of several OnHW-chars runs into an ensemble.

Each member is a ``--save-predictions`` file from ``imu2text.models``, trained
with the same ``--split-seed`` and a different ``--seed``, so every member
scores the same test samples in the same order. The ensemble prediction is
the argmax of the mean probability; nothing is selected or weighted, so the
test set is used for reporting only.

    python -m scripts.ensemble_chars --out results/ensemble/summary \\
        --group right 'results/ensemble/right_seed*.npz' \\
        --group both 'results/ensemble/both_seed*.npz'

Reported per group: accuracy for each member and for the ensemble, split by
handedness when the runs pooled both archives, case-insensitive accuracy,
negative log-likelihood and expected calibration error (ECE, 15 equal-width
confidence bins; Guo et al., "On Calibration of Modern Neural Networks",
ICML 2017).

Also per group, for the whole test set: how confident the model is on its
case errors (a letter read as its own other case) against its other errors,
and which errors abstaining on the least confident predictions removes. This
answers issue #13's question of whether the case confusions are confidently
wrong.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

ECE_BINS = 15
COVERAGES = (0.9, 0.8, 0.7)
POPULATIONS = (("all", None), ("right-handed", (0, -1)), ("left-handed", (1,)))


def load_members(pattern: str) -> list:
    """Load every .npz matching ``pattern`` and check that they line up."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no files match {pattern}")
    members = []
    for path in paths:
        with np.load(path, allow_pickle=False) as d:
            if "proba" not in d:
                raise SystemExit(
                    f"{path} has no probabilities; rerun with this version"
                )
            members.append({k: d[k] for k in d.files} | {"path": path})
    ref = members[0]
    for m in members[1:]:
        for key in ("true", "handedness", "classes", "split", "split_seed"):
            if not np.array_equal(m[key], ref[key]):
                raise SystemExit(
                    f"{m['path']} and {ref['path']} differ in '{key}'; members "
                    "must share one split (same --split-seed and data flags)"
                )
    seeds = [int(m["seed"]) for m in members]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"duplicate seeds in {pattern}: {seeds}")
    return members


def expected_calibration_error(proba: np.ndarray, true: np.ndarray) -> float:
    """Weighted mean gap between confidence and accuracy over confidence bins."""
    conf = proba.max(1)
    correct = proba.argmax(1) == true
    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    # Right-closed bins, so a confidence of exactly 1.0 lands in the last one.
    bins = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, ECE_BINS - 1)
    ece = 0.0
    for b in range(ECE_BINS):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def scores(proba: np.ndarray, true: np.ndarray, classes: np.ndarray) -> dict:
    """Accuracy, case-insensitive accuracy, NLL and ECE for one prediction set."""
    pred = proba.argmax(1)
    folded = np.char.lower(classes.astype(str))
    eps = 1e-12
    return {
        "n": int(len(true)),
        "accuracy": float((pred == true).mean() * 100),
        "case_insensitive_accuracy": float((folded[pred] == folded[true]).mean() * 100),
        "nll": float(-np.log(proba[np.arange(len(true)), true] + eps).mean()),
        "ece": float(expected_calibration_error(proba, true) * 100),
    }


def case_confidence(proba: np.ndarray, true: np.ndarray, classes: np.ndarray) -> dict:
    """Confidence of right answers, case errors and other errors, and abstention.

    A case error is a prediction that is the right letter in the other case.
    For each coverage c, the least confident (1 - c) of predictions are
    withheld; reported is the share of each kind of error withheld and the
    accuracy on what is kept.
    """
    pred, conf = proba.argmax(1), proba.max(1)
    folded = np.char.lower(classes.astype(str))
    right = pred == true
    case = ~right & (folded[pred] == folded[true])
    other = ~right & ~case
    second = np.argsort(-proba, axis=1)[:, 1]

    def describe(mask):
        c = conf[mask]
        return {
            "n": int(mask.sum()),
            "mean_confidence": float(c.mean()) if len(c) else 0.0,
            "median_confidence": float(np.median(c)) if len(c) else 0.0,
            "share_at_least_0_8": float((c >= 0.8).mean() * 100) if len(c) else 0.0,
            "share_at_least_0_9": float((c >= 0.9).mean() * 100) if len(c) else 0.0,
        }

    order = np.argsort(conf, kind="stable")
    abstain = []
    for cov in COVERAGES:
        held = np.zeros(len(conf), bool)
        held[order[: int(round(len(conf) * (1 - cov)))]] = True
        kept_errors = (~right & ~held).sum()
        abstain.append(
            {
                "coverage": cov,
                "accuracy_kept": float(right[~held].mean() * 100),
                "case_errors_withheld": float(held[case].mean() * 100),
                "other_errors_withheld": float(held[other].mean() * 100),
                "right_withheld": float(held[right].mean() * 100),
                "case_share_of_kept_errors": float(
                    (case & ~held).sum() / max(kept_errors, 1) * 100
                ),
            }
        )
    return {
        "right": describe(right),
        "case_errors": describe(case),
        "other_errors": describe(other),
        "case_share_of_errors": float(case.sum() / max((~right).sum(), 1) * 100),
        "case_errors_second_choice_right": float(
            (second[case] == true[case]).mean() * 100 if case.any() else 0.0
        ),
        "abstain": abstain,
    }


def by_population(proba, true, handedness, classes) -> dict:
    """Score the whole test set and each handedness population present."""
    out = {}
    for name, codes in POPULATIONS:
        mask = np.ones(len(true), bool) if codes is None else np.isin(handedness, codes)
        if mask.any():
            out[name] = scores(proba[mask], true[mask], classes)
    return out


def summarise(name: str, members: list) -> dict:
    """Per-member and ensemble scores for one group of runs."""
    ref = members[0]
    true, hand, classes = ref["true"], ref["handedness"], ref["classes"]
    per_member = [
        {
            "seed": int(m["seed"]),
            "val_acc": float(m["val_acc"]),
            **by_population(m["proba"], true, hand, classes),
        }
        for m in members
    ]
    mean_proba = np.mean([m["proba"] for m in members], axis=0)
    return {
        "group": name,
        "split": str(ref["split"]),
        "split_seed": int(ref["split_seed"]),
        "model": str(ref["model"]),
        "seeds": [int(m["seed"]) for m in members],
        "files": [os.path.basename(m["path"]) for m in members],
        "members": per_member,
        "ensemble": by_population(mean_proba, true, hand, classes),
        "case_confidence": {
            "ensemble": case_confidence(mean_proba, true, classes),
            "members": [case_confidence(m["proba"], true, classes) for m in members],
        },
    }


def markdown(groups: list) -> str:
    """One table row per group and population: member mean/range and ensemble."""
    lines = [
        "| Group | Test population | n | Members: mean (min-max) % | "
        "Ensemble % | Case-insensitive % | ECE % single → ensemble |",
        "|---|---|--:|--:|--:|--:|--:|",
    ]
    for g in groups:
        for pop, ens in g["ensemble"].items():
            accs = [m[pop]["accuracy"] for m in g["members"]]
            eces = [m[pop]["ece"] for m in g["members"]]
            lines.append(
                f"| {g['group']} ({len(accs)} seeds) | {pop} | {ens['n']} | "
                f"{np.mean(accs):.2f} ({min(accs):.2f}-{max(accs):.2f}) | "
                f"**{ens['accuracy']:.2f}** | {ens['case_insensitive_accuracy']:.2f} | "
                f"{np.mean(eces):.2f} → {ens['ece']:.2f} |"
            )
    splits = "; ".join(f"{g['group']}: {g['split']}" for g in groups)
    lines += ["", f"Splits: {splits}. Model: {groups[0]['model']}."]
    lines += ["", *case_markdown(groups)]
    return "\n".join(lines) + "\n"


def case_markdown(groups: list) -> list:
    """Case errors against other errors: confidence, and what abstaining removes."""
    lines = [
        "Case errors (a letter read as its own other case) against other errors, "
        "whole test set. Single model: mean over the members.",
        "",
        "| Group | Model | Kind | n | Median confidence | Confidence ≥ 0.9 |",
        "|---|---|---|--:|--:|--:|",
    ]
    for g in groups:
        for name, runs in (
            ("single", g["case_confidence"]["members"]),
            ("ensemble", [g["case_confidence"]["ensemble"]]),
        ):
            for kind, label in (
                ("right", "right"),
                ("case_errors", "case error"),
                ("other_errors", "other error"),
            ):
                n = np.mean([r[kind]["n"] for r in runs])
                med = np.mean([r[kind]["median_confidence"] for r in runs])
                hi = np.mean([r[kind]["share_at_least_0_9"] for r in runs])
                lines.append(
                    f"| {g['group']} | {name} | {label} | {n:.0f} | {med:.3f} | {hi:.1f}% |"
                )
    lines += [
        "",
        "Abstaining on the least confident predictions (ensemble):",
        "",
        "| Group | Coverage | Accuracy kept % | Case errors withheld % | "
        "Other errors withheld % | Right answers withheld % | Case share of kept errors % |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for g in groups:
        ens = g["case_confidence"]["ensemble"]
        lines.append(
            f"| {g['group']} | 100% | "
            f"{ens['right']['n'] / (ens['right']['n'] + ens['case_errors']['n'] + ens['other_errors']['n']) * 100:.2f} "
            f"| 0.0 | 0.0 | 0.0 | {ens['case_share_of_errors']:.1f} |"
        )
        for a in ens["abstain"]:
            lines.append(
                f"| {g['group']} | {a['coverage'] * 100:.0f}% | {a['accuracy_kept']:.2f} | "
                f"{a['case_errors_withheld']:.1f} | {a['other_errors_withheld']:.1f} | "
                f"{a['right_withheld']:.1f} | {a['case_share_of_kept_errors']:.1f} |"
            )
    lines += [
        "",
        "Case errors whose second choice is the right answer (ensemble): "
        + "; ".join(
            f"{g['group']} {g['case_confidence']['ensemble']['case_errors_second_choice_right']:.1f}%"
            for g in groups
        )
        + ".",
    ]
    return lines


def main() -> None:
    """CLI: summarise one or more groups of saved runs."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--group",
        nargs=2,
        action="append",
        required=True,
        metavar=("NAME", "GLOB"),
        help="a name and a quoted glob of --save-predictions files",
    )
    ap.add_argument("--out", required=True, help="output stem for .json and .md")
    args = ap.parse_args()

    groups = [summarise(name, load_members(pattern)) for name, pattern in args.group]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(f"{args.out}.json", "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2)
    table = markdown(groups)
    with open(f"{args.out}.md", "w", encoding="utf-8") as f:
        f.write(table)
    print(table)


if __name__ == "__main__":
    main()
