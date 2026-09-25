"""Calibration and selective-prediction figure for a single model vs an ensemble.

Reads the ``--save-predictions`` files that ``scripts/ensemble_chars.py``
averages, and draws two panels from the saved softmax outputs:

A. Reliability diagram: test accuracy against mean confidence in 15
   equal-width confidence bins, for one member and for the ensemble. A
   calibrated model sits on the diagonal.
B. Accuracy against coverage: keep only the most confident fraction of test
   samples and score those. The right edge (100% coverage) is the ordinary
   test accuracy; everything left of it answers "how accurate is the model
   on the predictions it is surest about".

    python -m scripts.plot_uncertainty 'results/ensemble/right_seed*.npz' \\
        --out results/uncertainty

Also writes ``<out>_curves.csv`` with every plotted point.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from scripts.ensemble_chars import ECE_BINS, expected_calibration_error, load_members
from scripts.plot_error_analysis import BLUE, GRAY, RED, TEXT, plt, style_axes

COVERAGE_MARKS = (100, 90, 80)


def reliability(proba, true):
    """Per-bin (mean confidence, accuracy, count) over equal-width bins."""
    conf = proba.max(1)
    correct = proba.argmax(1) == true
    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    bins = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, ECE_BINS - 1)
    rows = []
    for b in range(ECE_BINS):
        mask = bins == b
        if mask.any():
            rows.append((conf[mask].mean(), correct[mask].mean(), int(mask.sum())))
    return np.array(rows)


def coverage_curve(proba, true):
    """Accuracy of the k most confident samples, for k = 1..n, as percentages."""
    order = np.argsort(-proba.max(1), kind="stable")
    correct = (proba.argmax(1) == true)[order]
    k = np.arange(1, len(true) + 1)
    return 100 * k / len(true), 100 * np.cumsum(correct) / k


def main() -> None:  # pylint: disable=too-many-locals
    """CLI: draw the two panels for one group of ensemble members."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("members", help="quoted glob of --save-predictions files")
    ap.add_argument("--population", choices=["all", "right"], default="right")
    ap.add_argument("--out", default=os.path.join("results", "uncertainty"))
    args = ap.parse_args()

    members = load_members(args.members)
    ref = members[0]
    keep = (
        np.isin(ref["handedness"], (0, -1))
        if args.population == "right"
        else np.ones(len(ref["true"]), bool)
    )
    true = ref["true"][keep]
    single = members[0]["proba"][keep]
    ensemble = np.mean([m["proba"][keep] for m in members], axis=0)
    series = [
        (f"single model (seed {int(ref['seed'])})", single, RED, "o"),
        (f"{len(members)}-seed ensemble", ensemble, BLUE, "s"),
    ]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.2, 3.9))
    style_axes(ax_a, grid_axis="both")
    style_axes(ax_b, grid_axis="both")

    ax_a.plot(
        [0, 1], [0, 1], color=GRAY, lw=1, ls=(0, (3, 3)), zorder=1, label="calibrated"
    )
    csv_rows = []
    for name, proba, color, marker in series:
        rel = reliability(proba, true)
        ece = 100 * expected_calibration_error(proba, true)
        ax_a.plot(
            rel[:, 0],
            rel[:, 1],
            color=color,
            lw=1.6,
            marker=marker,
            ms=4.5,
            zorder=3,
            label=f"{name}, ECE {ece:.1f}%",
        )
        csv_rows += [("reliability", name, c, a, n) for c, a, n in rel]

        cov, acc = coverage_curve(proba, true)
        # The operating points go in the legend: labels inside the panel
        # always cross the rising curves.
        points = []
        for mark in COVERAGE_MARKS:
            i = int(round(mark / 100 * len(true))) - 1
            ax_b.plot(cov[i], acc[i], marker=marker, ms=5, color=color, zorder=4)
            csv_rows.append(("coverage", name, cov[i], acc[i], i + 1))
            points.append(f"{acc[i]:.1f}% at {mark}%")
        ax_b.plot(
            cov,
            acc,
            color=color,
            lw=1.6,
            zorder=3,
            label=f"{name}: {', '.join(points)}",
        )
    ax_a.set(xlim=(0, 1), ylim=(0, 1))
    ax_a.set_xlabel("mean confidence in bin", fontsize=8, color=TEXT)
    ax_a.set_ylabel("accuracy in bin", fontsize=8, color=TEXT)
    ax_a.set_title("A  Reliability", loc="left", fontsize=9, color=TEXT)
    ax_a.legend(frameon=False, fontsize=7, loc="upper left")

    ax_b.set(xlim=(100, 20))
    ax_b.set_ylim(top=100.5)
    ax_b.set_xlabel(
        "coverage: % of test samples kept (most confident first)", fontsize=8
    )
    ax_b.set_ylabel("accuracy on kept samples (%)", fontsize=8, color=TEXT)
    ax_b.set_title("B  Accuracy vs coverage", loc="left", fontsize=9, color=TEXT)
    ax_b.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.2))

    population = "right-handed" if args.population == "right" else "all"
    fig.suptitle(
        f"OnHW-chars {len(ref['classes'])}-class, {ref['split']}\n"
        f"Calibration and selective accuracy, {population} test samples "
        f"(n={len(true):,})",
        fontsize=9.5,
        color=TEXT,
        x=0.012,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            f"{args.out}.{ext}", dpi=200, bbox_inches="tight", facecolor="white"
        )
        print(f"wrote {args.out}.{ext}")
    with open(f"{args.out}_curves.csv", "w", encoding="utf-8") as f:
        f.write("panel,series,x,y,n\n")
        for panel, name, x, y, n in csv_rows:
            f.write(f"{panel},{name},{x:.4f},{y:.4f},{n}\n")
    print(f"wrote {args.out}_curves.csv")


if __name__ == "__main__":
    main()
