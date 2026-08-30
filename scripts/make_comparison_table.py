"""Run the six official OnHW-chars cells and tabulate them against the paper.

The published comparison is Table 3 of Ott et al., "Domain Adaptation for
Time-Series Classification to Mitigate Covariate Shift" (ACM MM 2022), which
reports CRR for right-handed writers across {lower, upper, combined} x
{writer-dependent, writer-independent}. The OnHW-chars archive ships exactly
those six splits, so the same table can be produced here and the numbers put
side by side.

Which published number to compare against matters more than it looks. Table 4
of the same paper reports figures up to 100.00, but that table is a *domain
adaptation* benchmark: a right-handed model carried onto left-handed writers
using labelled samples from those writers, scored on a small left-handed
validation set whose baseline is 25.19. It is not writer-independent character
recognition and is not comparable to anything here. Table 3's right-handed WI
column is.

Usage
-----
    python scripts/make_comparison_table.py --epochs 30
    python scripts/make_comparison_table.py --config best --epochs 30
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CASES = ("lower", "upper", "both")
DEPENDENCIES = ("dep", "indep")

# Ott et al., ACM MM 2022, Table 3, right-handed writers (U_Tv). CRR in %.
# Transcribed from data/ACMMM_2022.pdf; these are their numbers, not ours.
PUBLISHED = {
    "CNN+BiLSTM [60]": {
        ("lower", "dep"): 88.85,
        ("lower", "indep"): 79.48,
        ("upper", "dep"): 92.15,
        ("upper", "indep"): 85.60,
        ("both", "dep"): 78.17,
        ("both", "indep"): 68.06,
    },
    "InceptionTime [25]": {
        ("lower", "dep"): 84.14,
        ("lower", "indep"): 75.28,
        ("upper", "dep"): 87.80,
        ("upper", "indep"): 81.62,
        ("both", "dep"): 70.43,
        ("both", "indep"): 61.68,
    },
    "ResNet [86]": {
        ("lower", "dep"): 83.01,
        ("lower", "indep"): 71.93,
        ("upper", "dep"): 86.41,
        ("upper", "indep"): 78.03,
        ("both", "dep"): 68.56,
        ("both", "indep"): 58.74,
    },
    "LSTM-FCN [45]": {
        ("lower", "dep"): 81.43,
        ("lower", "indep"): 71.41,
        ("upper", "dep"): 85.43,
        ("upper", "indep"): 77.07,
        ("both", "dep"): 67.34,
        ("both", "indep"): 57.93,
    },
}

CONFIGS = {
    "baseline": {
        "label": "CNN+BiLSTM (this repo)",
        "args": ["--models", "cnn_bilstm"],
    },
    "best": {
        "label": "CNN+BiLSTM+attn, aug x2, LS, LR sched (this repo)",
        "args": [
            "--models",
            "cnn_bilstm_attn",
            "--augment",
            "2",
            "--aug-policy",
            "extended",
            "--label-smoothing",
            "0.1",
            "--lr-schedule",
        ],
    },
}


def run_cell(chars_dir, case, dependency, fold, epochs, seed, extra, deterministic):
    """Train one split and return its held-out accuracy, or None if it failed."""
    cmd = [
        sys.executable,
        "-m",
        "imu2text.models",
        "--onhw-chars",
        chars_dir,
        "--case",
        case,
        "--dependency",
        dependency,
        "--fold",
        str(fold),
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
    ] + extra
    if deterministic:
        cmd.append("--deterministic")
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if proc.returncode != 0:
        print(f"  {case}/{dependency}: FAILED\n{proc.stderr[-600:]}")
        return None
    for line in proc.stdout.splitlines():
        if "Best held-out:" in line:
            acc = float(line.split("@")[1].split("%")[0])
            print(f"  {case}/{dependency}: {acc:.2f}%  ({time.time() - started:.0f}s)")
            return acc
    print(f"  {case}/{dependency}: no result line found")
    return None


def render(results, label, out_md):
    """Write the markdown table, ours beside the published rows."""
    header = (
        "| Method | Lower WD | Lower WI | Upper WD | Upper WI "
        "| Combined WD | Combined WI |\n|---|--:|--:|--:|--:|--:|--:|\n"
    )
    rows = ""
    for name, cells in PUBLISHED.items():
        vals = " | ".join(f"{cells[(c, d)]:.2f}" for c in CASES for d in DEPENDENCIES)
        rows += f"| {name} | {vals} |\n"
    ours = " | ".join(
        (f"{results[(c, d)]:.2f}" if results.get((c, d)) is not None else "-")
        for c in CASES
        for d in DEPENDENCIES
    )
    rows += f"| **{label}** | {ours} |\n"

    text = (
        "Published rows are Ott et al., ACM MM 2022, Table 3, right-handed\n"
        "writers. Our row is the same six official OnHW-chars splits, fold 0,\n"
        "single seed.\n\n" + header + rows
    )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"wrote {out_md}")


def main():
    """Sweep the six official splits and tabulate against the published table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onhw-chars", default="data/onhw-chars_2021-06-30")
    ap.add_argument("--config", choices=sorted(CONFIGS), default="baseline")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    print(
        f"{cfg['label']}: {len(CASES) * len(DEPENDENCIES)} cells, "
        f"{args.epochs} epochs each"
    )

    results = {}
    for case in CASES:
        for dependency in DEPENDENCIES:
            results[(case, dependency)] = run_cell(
                args.onhw_chars,
                case,
                dependency,
                args.fold,
                args.epochs,
                args.seed,
                cfg["args"],
                args.deterministic,
            )

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"comparison_{args.config}")
    with open(f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": args.config,
                "label": cfg["label"],
                "epochs": args.epochs,
                "fold": args.fold,
                "seed": args.seed,
                "deterministic": args.deterministic,
                "results": {f"{c}_{d}": v for (c, d), v in results.items()},
            },
            f,
            indent=2,
        )
    render(results, cfg["label"], f"{stem}.md")


if __name__ == "__main__":
    main()
