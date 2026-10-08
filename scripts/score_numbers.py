"""Score saved OnHW-symbols or split-equations predictions on numbers.

Reads a ``--save-predictions`` file from ``imu2text.models`` trained with
``--onhw-symbols``. For every file it reports accuracy on all 15 symbols, on
the ten digits alone and on the five operators, and per class. Given the
archive directory of an equations run (``--symbols-kind equations``), it also
puts the slices back into their equations with
``imu2text.symbols.load_equation_groups`` and reports:

- number accuracy: a number is a maximal run of digits inside one equation,
  read off the true labels, and it counts as right only if every digit in it
  is right;
- equation accuracy: every symbol of the equation right.

The slices are cut by the dataset authors, so these are scores for a
recogniser given the true segmentation, not for one that has to find it.

    python -m scripts.score_numbers \\
        --task symbols_indep=results/tasks/symbols_indep.npz \\
        --task equations_indep=results/tasks/equations_indep.npz:data/OnHW-symbols_equations_indep \\
        --out results/tasks/numbers
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from imu2text.symbols import SYMBOLS_VOCAB, load_equation_groups

N_DIGITS = 10


def pct(hits: np.ndarray) -> float:
    """Percentage of true values, rounded to two decimals."""
    return round(100.0 * float(np.mean(hits)), 2) if len(hits) else float("nan")


def number_runs(true: np.ndarray, groups: np.ndarray) -> list:
    """Index arrays of each maximal digit run inside one equation."""
    runs, current = [], []
    for i, (label, group) in enumerate(zip(true, groups)):
        if current and (label >= N_DIGITS or group != groups[current[-1]]):
            runs.append(np.asarray(current))
            current = []
        if label < N_DIGITS:
            current.append(i)
    if current:
        runs.append(np.asarray(current))
    return runs


def score(true: np.ndarray, pred: np.ndarray, groups=None) -> dict:
    """Symbol, digit, operator and per-class accuracy; numbers and equations
    as well when ``groups`` maps each sample to its equation."""
    hit = true == pred
    digit = true < N_DIGITS
    out = {
        "n_symbols": int(len(true)),
        "symbol_acc": pct(hit),
        "n_digits": int(digit.sum()),
        "digit_acc": pct(hit[digit]),
        "n_operators": int((~digit).sum()),
        "operator_acc": pct(hit[~digit]),
        "digit_read_as_operator": int((digit & (pred >= N_DIGITS)).sum()),
        "per_class": {
            SYMBOLS_VOCAB[k]: {"n": int((true == k).sum()), "acc": pct(hit[true == k])}
            for k in range(len(SYMBOLS_VOCAB))
        },
    }
    if groups is not None:
        if len(groups) != len(true):
            raise SystemExit(
                f"{len(groups)} equation ids for {len(true)} predictions; the "
                "archive does not match the run"
            )
        runs = number_runs(true, groups)
        lengths = np.asarray([len(r) for r in runs])
        run_hit = np.asarray([hit[r].all() for r in runs])
        out["n_numbers"] = len(runs)
        out["number_acc"] = pct(run_hit)
        out["number_acc_by_length"] = {
            str(n) if n < 4 else "4+": {
                "n": int(sel.sum()),
                "acc": pct(run_hit[sel]),
            }
            for n, sel in (
                (n, (lengths == n) if n < 4 else (lengths >= 4)) for n in (1, 2, 3, 4)
            )
        }
        ids = np.unique(groups)
        eq_hit = np.asarray([hit[groups == g].all() for g in ids])
        out["n_equations"] = int(len(ids))
        out["equation_acc"] = pct(eq_hit)
    return out


def load_task(spec: str) -> tuple:
    """Parse ``name=predictions.npz[:archive_dir]``."""
    name, _, rest = spec.partition("=")
    path, _, archive = rest.partition(":")
    if not name or not path:
        raise SystemExit(f"--task wants name=file.npz[:archive], got {spec!r}")
    with np.load(path, allow_pickle=False) as d:
        if "".join(d["classes"].tolist()) != SYMBOLS_VOCAB:
            raise SystemExit(f"{path} is not a 15-symbol run")
        run = {k: d[k] for k in ("true", "pred", "seed", "split", "test_acc")}
    groups = load_equation_groups(archive, "val") if archive else None
    return name, run, groups


def to_markdown(rows: dict) -> str:
    """One summary table across tasks."""
    lines = [
        "| Task | Split | Symbols | Digits | Operators | Numbers | Equations |",
        "|---|---|--:|--:|--:|--:|--:|",
    ]
    for name, r in rows.items():
        numbers = (
            f"{r['number_acc']:.2f} ({r['n_numbers']})" if "n_numbers" in r else "n/a"
        )
        equations = (
            f"{r['equation_acc']:.2f} ({r['n_equations']})"
            if "n_equations" in r
            else "n/a"
        )
        lines.append(
            f"| {name} | {r['split']}, seed {r['seed']} "
            f"| {r['symbol_acc']:.2f} ({r['n_symbols']}) "
            f"| {r['digit_acc']:.2f} ({r['n_digits']}) "
            f"| {r['operator_acc']:.2f} ({r['n_operators']}) "
            f"| {numbers} | {equations} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--task",
        action="append",
        required=True,
        metavar="NAME=NPZ[:ARCHIVE]",
        help="a saved run; add the archive directory to score equations",
    )
    ap.add_argument("--out", help="write OUT.json and OUT.md")
    args = ap.parse_args()

    rows = {}
    for spec in args.task:
        name, run, groups = load_task(spec)
        rows[name] = score(run["true"], run["pred"], groups) | {
            "seed": int(run["seed"]),
            "split": str(run["split"]),
        }
        if abs(rows[name]["symbol_acc"] - float(run["test_acc"])) > 0.01:
            raise SystemExit(f"{name}: rescored accuracy disagrees with the run")
    table = to_markdown(rows)
    print(table)
    if args.out:
        with open(args.out + ".json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
            f.write("\n")
        with open(args.out + ".md", "w", encoding="utf-8") as f:
            f.write(table)


if __name__ == "__main__":
    main()
