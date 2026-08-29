"""Publication-quality matplotlib figures for the OnHW benchmark results.

Rebuilds the plotting approach of ImpAcX_OnHW's ``plot_kNN_results.py``
(KorayKarabina/ImpAcX_OnHW) for this repository's results:

- the same LaTeX-friendly figure sizing (``set_size``: point width -> inches,
  golden-ratio height, from https://jwalton.info/Embed-Publication-Matplotlib-Latex/),
- mean-accuracy curves with an optional +/- std cloud (``fill_between``),
- vector PDF output with ``bbox_inches='tight'`` for direct LaTeX embedding.

Differences from the original (see docs/impacx_onhw_analysis.md): a fixed,
colorblind-safe palette instead of ad-hoc named CSS colors, and results are
read from CSV rather than per-fold pickle files.

Figures produced (into results/):
  1. learning_curve.pdf/.png - writer-independent accuracy vs. number of
     training writers, with the logistic projection fit (same model as
     onhw_projection.m) extrapolated to full-dataset scale.
  2. model_benchmarks.pdf/.png - held-out WI accuracy per architecture.

Usage:  python plot_results.py [--width 500]
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
# pylint: disable=wrong-import-position
# The backend has to be chosen before pyplot is imported, so these two cannot
# move to the top of the file.
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

# pylint: enable=wrong-import-position

# Colorblind-safe slots (validated palette; identity is never color-alone -
# every series is also direct-labeled).
BLUE = "#2a78d6"  # measured data
GRAY = "#52514e"  # fitted / reference lines
RED = "#e34948"  # projection marker
TEXT = "#0b0b0b"
MUTED = "#83827d"

RESULTS_DIR = "results"
CURVE_CSV = os.path.join(RESULTS_DIR, "learning_curve.csv")

# Writer-independent benchmarks from README (bundled subset, seed 0);
# overridden by results/benchmarks.csv (columns: model,wi_test_acc) if present.
DEFAULT_BENCHMARKS = [
    ("cnn_bilstm", 64.8),
    ("bilstm", 56.2),
    ("cnn", 48.7),
    ("lstm", 43.8),
    ("majority baseline", 2.2),
]
FULL_SCALE_WRITERS = 71  # ~full OnHW-chars dataset at a 60% train share


def set_size(width: float, fraction: float = 1.0) -> tuple:
    """Figure dimensions (inches) for a LaTeX column width given in points.

    Same helper as ImpAcX_OnHW / https://jwalton.info/Embed-Publication-Matplotlib-Latex/:
    avoids rescaling in LaTeX (which would shrink fonts) by sizing the figure
    to the document's textwidth, with golden-ratio height.
    """
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    return fig_width_in, fig_width_in * golden_ratio


def style_axes(ax) -> None:
    """Recessive axes: no top/right spines, light y-grid behind the data."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT)
    ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)


def logistic(w, L, a, w0):
    """acc(W) = L / (1 + exp(-a (W - w0))) - the onhw_projection.m model."""
    return L / (1.0 + np.exp(-a * (w - w0)))


def plot_learning_curve(width: float) -> None:
    if not os.path.exists(CURVE_CSV):
        print(
            f"skip learning-curve figure ({CURVE_CSV} not found; "
            f"run make_learning_curve.py first)"
        )
        return
    with open(CURVE_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    writers = np.array([float(r["n_writers"]) for r in rows])
    acc = np.array([float(r["wi_test_acc"]) for r in rows])

    fig, ax = plt.subplots(1, 1, figsize=set_size(width))
    style_axes(ax)

    # Logistic fit + extrapolation to full-dataset writer count
    try:
        (L, a, w0), _ = curve_fit(
            logistic, writers, acc, p0=[80.0, 0.15, 5.0], maxfev=20000
        )
        w_grid = np.linspace(1, FULL_SCALE_WRITERS, 200)
        ax.plot(
            w_grid,
            logistic(w_grid, L, a, w0),
            "--",
            color=GRAY,
            linewidth=1.2,
            label="logistic fit",
        )
        proj = logistic(FULL_SCALE_WRITERS, L, a, w0)
        ax.plot([FULL_SCALE_WRITERS], [proj], "o", color=RED, markersize=5)
        ax.annotate(
            f"projected {proj:.1f}%\n@ {FULL_SCALE_WRITERS} writers",
            (FULL_SCALE_WRITERS, proj),
            textcoords="offset points",
            xytext=(-8, -24),
            ha="right",
            fontsize=8,
            color=TEXT,
        )
        ax.axhline(L, color=MUTED, linewidth=0.6, linestyle=":")
        ax.annotate(
            f"ceiling L = {L:.1f}%",
            (1, L),
            textcoords="offset points",
            xytext=(2, 4),
            fontsize=8,
            color=MUTED,
        )
    except RuntimeError:
        print("logistic fit did not converge; plotting measurements only")

    ax.plot(
        writers,
        acc,
        "-o",
        color=BLUE,
        linewidth=2,
        markersize=5,
        label="measured (CNN+BiLSTM)",
    )
    for wx, ay in zip(writers, acc):
        ax.annotate(
            f"{ay:.1f}",
            (wx, ay),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7.5,
            color=TEXT,
        )

    ax.set_xlabel("Training writers (writer-independent split)")
    ax.set_ylabel("WI test accuracy (%)")
    ax.set_title("OnHW-chars learning curve and projection")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(RESULTS_DIR, f"learning_curve.{ext}"),
            format=ext,
            bbox_inches="tight",
            dpi=200,
        )
    plt.close(fig)
    print("wrote results/learning_curve.pdf/.png")


def load_benchmarks():
    path = os.path.join(RESULTS_DIR, "benchmarks.csv")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return [(r["model"], float(r["wi_test_acc"])) for r in rows]
    return DEFAULT_BENCHMARKS


def plot_benchmarks(width: float) -> None:
    data = sorted(load_benchmarks(), key=lambda r: r[1])
    names = [d[0] for d in data]
    accs = [d[1] for d in data]

    fig, ax = plt.subplots(1, 1, figsize=set_size(width))
    style_axes(ax)
    ax.grid(False)
    ax.grid(axis="x", color=MUTED, alpha=0.25, linewidth=0.5)

    y = np.arange(len(names))
    ax.barh(y, accs, height=0.55, color=BLUE)
    for yi, a in zip(y, accs):
        ax.annotate(
            f"{a:.1f}%",
            (a, yi),
            textcoords="offset points",
            xytext=(4, 0),
            va="center",
            fontsize=8,
            color=TEXT,
        )
    ax.set_yticks(y, names)
    ax.set_xlabel("Writer-independent test accuracy (%)")
    ax.set_title("OnHW-chars: held-out accuracy by architecture (52 classes)")
    ax.set_xlim(0, 100)

    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(RESULTS_DIR, f"model_benchmarks.{ext}"),
            format=ext,
            bbox_inches="tight",
            dpi=200,
        )
    plt.close(fig)
    print("wrote results/model_benchmarks.pdf/.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--width",
        type=float,
        default=500.0,
        help="target document width in points (LaTeX \\the\\textwidth)",
    )
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_learning_curve(args.width)
    plot_benchmarks(args.width)


if __name__ == "__main__":
    main()
