"""Compare two configurations run over the same seeds, seed by seed.

Each argument is ``name=glob`` over ``--save-predictions`` files from
``imu2text.models``, named ``..._seed<N>.npz``. Runs are paired by seed; a
pair must share the test labels, class order, split label and split seed,
so the only difference inside a pair is the configuration. Reported per
seed: test, case-insensitive and validation accuracy; then the mean and
sample standard deviation of each, the paired difference with its t
statistic (n - 1 degrees of freedom), and the accuracy of each
configuration's mean-probability vote over its seeds.

    python -m scripts.compare_seeds --out results/masking/summary \\
        plain='results/masking/plain_seed*.npz' \\
        masked='results/masking/masked_seed*.npz'
"""

from __future__ import annotations

import argparse
import glob
import json
import re

import numpy as np

PAIRED_KEYS = ("true", "classes", "split", "split_seed")


def load(pattern: str) -> dict:
    """Map seed -> run, from files matching ``pattern``."""
    runs = {}
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"seed(\d+)\.npz$", path)
        if not m:
            raise SystemExit(f"{path}: expected a name ending in seed<N>.npz")
        with np.load(path, allow_pickle=False) as d:
            runs[int(m.group(1))] = {k: d[k] for k in d.files} | {"path": path}
    if not runs:
        raise SystemExit(f"no files match {pattern}")
    return runs


def scores(run: dict) -> dict:
    """Test, case-insensitive and validation accuracy of one run, in percent."""
    true, pred, classes = run["true"], run["pred"], run["classes"]
    lower = np.char.lower(classes.astype(str))
    return {
        "test": float(100 * np.mean(true == pred)),
        "case_insensitive": float(100 * np.mean(lower[true] == lower[pred])),
        "val": float(run["val_acc"]),
        "train": float(run["train_acc"]),
    }


def compare(a: dict, b: dict) -> dict:
    """Paired comparison of two seed -> run maps."""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        raise SystemExit(f"need two or more shared seeds, got {seeds}")
    for s in seeds:
        for key in PAIRED_KEYS:
            if not np.array_equal(a[s][key], b[s][key]):
                raise SystemExit(f"seed {s}: the runs differ in '{key}'")
    rows = [{"seed": s, "a": scores(a[s]), "b": scores(b[s])} for s in seeds]
    diff = np.array([r["b"]["test"] - r["a"]["test"] for r in rows])
    out = {"seeds": seeds, "rows": rows}
    for side, runs in (("a", a), ("b", b)):
        vals = {k: np.array([r[side][k] for r in rows]) for k in rows[0][side]}
        vote = np.mean([runs[s]["proba"] for s in seeds], axis=0).argmax(axis=1)
        out[side] = {
            k: {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
            for k, v in vals.items()
        }
        out[side]["vote"] = float(100 * np.mean(vote == runs[seeds[0]]["true"]))
    out["difference"] = {
        "per_seed": diff.round(2).tolist(),
        "mean": float(diff.mean()),
        "sd": float(diff.std(ddof=1)),
        "t": float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))),
        "df": len(diff) - 1,
        "b_better": int(np.sum(diff > 0)),
    }
    return out


def to_markdown(name_a: str, name_b: str, res: dict, split: str) -> str:
    """A per-seed table and a summary line."""
    lines = [
        f"| Seed | {name_a} test % | {name_b} test % | Difference | {name_a} case-insensitive % | {name_b} case-insensitive % |",
        "|--:|--:|--:|--:|--:|--:|",
    ]
    for r, d in zip(res["rows"], res["difference"]["per_seed"]):
        lines.append(
            f"| {r['seed']} | {r['a']['test']:.2f} | {r['b']['test']:.2f} | {d:+.2f} "
            f"| {r['a']['case_insensitive']:.2f} | {r['b']['case_insensitive']:.2f} |"
        )
    a, b, d = res["a"], res["b"], res["difference"]
    lines.append(
        f"| mean | {a['test']['mean']:.2f} | {b['test']['mean']:.2f} | {d['mean']:+.2f} "
        f"| {a['case_insensitive']['mean']:.2f} | {b['case_insensitive']['mean']:.2f} |"
    )
    lines += [
        "",
        f"{split}. Paired difference {d['mean']:+.2f} points (sd {d['sd']:.2f}, "
        f"t = {d['t']:.2f} on {d['df']} df); {name_b} ahead on {d['b_better']} of "
        f"{len(res['seeds'])} seeds. Mean-probability vote over the seeds: "
        f"{name_a} {a['vote']:.2f}%, {name_b} {b['vote']:.2f}%.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("a", metavar="NAME=GLOB")
    ap.add_argument("b", metavar="NAME=GLOB")
    ap.add_argument("--out", help="write OUT.json and OUT.md")
    args = ap.parse_args()
    (name_a, pat_a), (name_b, pat_b) = (x.split("=", 1) for x in (args.a, args.b))
    runs_a, runs_b = load(pat_a), load(pat_b)
    res = compare(runs_a, runs_b)
    split = str(runs_a[res["seeds"][0]]["split"])
    table = to_markdown(name_a, name_b, res, split)
    print(table)
    if args.out:
        with open(args.out + ".json", "w", encoding="utf-8") as f:
            json.dump({"a": name_a, "b": name_b, "split": split} | res, f, indent=2)
            f.write("\n")
        with open(args.out + ".md", "w", encoding="utf-8") as f:
            f.write(table)


if __name__ == "__main__":
    main()
