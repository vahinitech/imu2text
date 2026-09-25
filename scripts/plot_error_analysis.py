"""Figure: where the OnHW-chars errors are, and why the sensor cannot fix them.

The 52-class benchmark stopped responding to modelling effort at about 72%.
This figure is the evidence for why: the task is 26 letters in two cases, and
for the letters whose two cases share a glyph shape the only distinguishing
cue is size, which an IMU never observes.

Four panels:

A. Confusion matrix, row-normalised, diagonal removed. Only the errors are
   drawn. Labels run A-Z then a-z, so a letter confused with its own other
   case lands exactly 26 off the diagonal - the two bright lines.
B. The most common confusions, coloured by whether they are a case pair.
C. Per-letter accuracy scored two ways: as 52 classes, and with case folded
   away. The gap is what case costs that letter.
D. How separable the two cases actually are, as AUC over the test set. 0.5 is
   a coin flip. Needs the dataset, so it is drawn only when --onhw-chars is
   given.

Usage
-----
    python -m imu2text.models --models cnn_bilstm_attn \\
        --onhw-chars data/onhw-chars_2021-06-30 --case both \\
        --dependency indep --fold 0 --epochs 30 \\
        --augment 2 --aug-policy extended --label-smoothing 0.1 \\
        --lr-schedule --save-predictions results/predictions_official_fold0.npz

    python scripts/plot_error_analysis.py \\
        --predictions results/predictions_official_fold0.npz \\
        --onhw-chars data/onhw-chars_2021-06-30
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
# pylint: disable=wrong-import-position
# The backend has to be chosen before pyplot is imported.
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pylint: enable=wrong-import-position

# Same validated slots as scripts/plot_results.py. Two categorical hues only
# (case pair / other), each also carried by its position and a direct label,
# so identity is never colour-alone.
BLUE = "#2a78d6"
RED = "#e34948"
GRAY = "#52514e"
TEXT = "#0b0b0b"
MUTED = "#83827d"
GRID = "#e6e5e2"

# Sequential ramp for the confusion matrix: one hue, light to dark. Magnitude
# is the job there, so a rainbow or a second hue would be wrong.
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f7fc", BLUE, "#123a68"])

# The ten letters whose upper and lower forms are the same shape at different
# sizes. J/j is excluded: the descender changes the stroke.
SAME_SHAPE = "COSUVWXZKP"

RESULTS_DIR = "results"


def style_axes(ax, grid_axis="y"):
    """Recessive frame: no top/right spines, hairline grid behind the data."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=7, length=0)


def rank_auc(a, b):
    """P(a random draw from `a` exceeds one from `b`), via the rank statistic.

    0.5 means the two distributions are indistinguishable by this feature, so
    the value reads directly as "can this cue separate the two cases at all".
    """
    if len(a) == 0 or len(b) == 0:
        return np.nan
    joined = np.concatenate([a, b])
    ranks = joined.argsort().argsort() + 1
    return (ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def panel_confusion(ax, true, pred, classes):
    """Row-normalised confusion, diagonal removed so only errors are drawn."""
    n = len(classes)
    cm = np.zeros((n, n))
    for t, p in zip(true, pred):
        cm[t, p] += 1
    with np.errstate(invalid="ignore"):
        cm = cm / cm.sum(axis=1, keepdims=True)
    np.fill_diagonal(cm, np.nan)  # correct predictions are not the subject

    im = ax.imshow(cm, cmap=SEQ, vmin=0, vmax=np.nanmax(cm), interpolation="nearest")
    ax.set_xlabel("predicted", fontsize=8, color=TEXT)
    ax.set_ylabel("true", fontsize=8, color=TEXT)
    ticks = [0, 12, 25, 38, 51]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([classes[i] for i in ticks])
    ax.set_yticklabels([classes[i] for i in ticks])
    ax.tick_params(colors=MUTED, labelsize=7, length=0)
    for side in ax.spines.values():
        side.set_color(GRID)

    half = n // 2
    # The case-pair locus: true i, predicted i +/- 26.
    ax.plot([half, n - 1], [0, half - 1], color=RED, lw=1.0, alpha=0.9)
    ax.plot([0, half - 1], [half, n - 1], color=RED, lw=1.0, alpha=0.9)
    ax.annotate(
        "same letter,\nother case",
        xy=(half + 6, 6),
        xytext=(half - 2, 17),
        fontsize=7,
        color=RED,
        ha="center",
        arrowprops={"arrowstyle": "-", "color": RED, "lw": 0.8},
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("share of that letter's samples", fontsize=7, color=MUTED)
    cbar.ax.tick_params(colors=MUTED, labelsize=6, length=0)
    cbar.outline.set_edgecolor(GRID)
    ax.set_title("A  Where the errors go", fontsize=9, color=TEXT, loc="left", pad=8)


def panel_top_confusions(ax, true, pred, classes, top=12):
    """The most common confusions, split by whether they are a case pair."""
    wrong = true != pred
    pairs = {}
    for t, p in zip(true[wrong], pred[wrong]):
        pairs[(int(t), int(p))] = pairs.get((int(t), int(p)), 0) + 1
    ranked = sorted(pairs.items(), key=lambda kv: -kv[1])[:top][::-1]

    n_err = int(wrong.sum())
    labels = [f"{classes[t]} → {classes[p]}" for (t, p), _ in ranked]
    counts = [c for _, c in ranked]
    is_case = [classes[t].lower() == classes[p].lower() for (t, p), _ in ranked]
    colors = [RED if c else BLUE for c in is_case]

    ypos = np.arange(len(ranked))
    ax.barh(ypos, counts, color=colors, height=0.62, zorder=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=7, fontfamily="monospace")
    ax.set_xlabel("test samples", fontsize=8, color=TEXT)
    style_axes(ax, grid_axis="x")

    for y, count in zip(ypos, counts):
        ax.text(
            count + max(counts) * 0.015,
            y,
            f"{100 * count / n_err:.1f}%",
            va="center",
            fontsize=6.5,
            color=MUTED,
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=RED),
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
    ]
    ax.set_xlim(0, max(counts) * 1.14)
    ax.legend(
        handles,
        ["same letter, other case", "a different letter"],
        fontsize=7,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, -0.13),
        ncol=2,
        labelcolor=TEXT,
        handlelength=1.1,
        columnspacing=1.2,
    )
    share = (
        100
        * sum(
            c for (t, p), c in pairs.items() if classes[t].lower() == classes[p].lower()
        )
        / n_err
    )
    ax.set_title(
        f"B  Top {top} confusions ({share:.0f}% of all errors are case)",
        fontsize=9,
        color=TEXT,
        loc="left",
        pad=8,
    )


def panel_per_letter(ax, true, pred, classes):
    """Per-letter accuracy scored as 52 classes, and with case folded away."""
    letters = sorted({c.lower() for c in classes})
    folded_true = np.array([classes[t].lower() for t in true])
    folded_pred = np.array([classes[p].lower() for p in pred])

    strict, folded = [], []
    for ch in letters:
        mask = folded_true == ch
        strict.append(100 * (true[mask] == pred[mask]).mean())
        folded.append(100 * (folded_pred[mask] == ch).mean())

    order = np.argsort(strict)
    letters = [letters[i] for i in order]
    strict = np.array(strict)[order]
    folded = np.array(folded)[order]

    xpos = np.arange(len(letters))
    ax.vlines(xpos, strict, folded, color=GRID, lw=2.4, zorder=1)
    ax.scatter(xpos, folded, s=22, color=BLUE, zorder=3, label="case folded away")
    ax.scatter(xpos, strict, s=22, color=RED, zorder=3, label="as 52 classes")
    ax.set_xticks(xpos)
    ax.set_xticklabels(letters, fontsize=7, fontfamily="monospace")
    ax.set_ylabel("accuracy (%)", fontsize=8, color=TEXT)
    ax.set_ylim(0, 105)
    style_axes(ax)
    ax.legend(
        fontsize=7,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.005),
        ncol=2,
        labelcolor=TEXT,
        handlelength=1.1,
        columnspacing=1.2,
    )

    worst = int(np.argmax(folded - strict))
    ax.annotate(
        f"{letters[worst]}: {folded[worst] - strict[worst]:.0f} points to case",
        xy=(xpos[worst], folded[worst] + 3.5),
        fontsize=7,
        color=TEXT,
        va="bottom",
        ha="center",
    )
    ax.set_ylim(min(strict) - 12, 108)
    ax.set_title(
        "C  What case costs each letter", fontsize=9, color=TEXT, loc="left", pad=8
    )


def panel_separability(ax, chars_dir, case, dependency, fold):
    """Can any cue in the signal tell the two cases apart? AUC, 0.5 = no."""
    from imu2text.chars import load_onhw_chars  # noqa: PLC0415

    ds = load_onhw_chars(chars_dir, case=case, dependency=dependency, fold=fold)
    y = ds.y_all
    classes = list(ds.classes)
    lens = np.array([len(s) for s in ds.X_all], dtype=float)
    rms = np.array(
        [
            (
                np.sqrt((np.asarray(s, dtype=np.float64)[:, 0:3] ** 2).sum(1)).mean()
                if len(s)
                else np.nan
            )
            for s in ds.X_all
        ]
    )

    groups = {
        "same shape\n(C/c, O/o, S/s, ...)": SAME_SHAPE,
        "different shape\n(A/a, E/e, R/r, ...)": "AERBHGDQ",
    }
    width = 0.34
    xpos = np.arange(len(groups))
    for offset, (feature, values, color, label) in enumerate(
        [(0, rms, BLUE, "acceleration RMS"), (1, lens, RED, "duration")]
    ):
        del feature
        heights = []
        for chars in groups.values():
            aucs = [
                rank_auc(
                    values[y == classes.index(ch)],
                    values[y == classes.index(ch.lower())],
                )
                for ch in chars
                if ch in classes and ch.lower() in classes
            ]
            heights.append(np.nanmean(aucs))
        ax.bar(
            xpos + (offset - 0.5) * width,
            heights,
            width * 0.92,
            color=color,
            label=label,
            zorder=2,
        )
        for x, h in zip(xpos + (offset - 0.5) * width, heights):
            ax.text(x, h + 0.012, f"{h:.2f}", ha="center", fontsize=7, color=MUTED)

    # 0.5 is a real threshold here, not a grid line, so a dashed rule is right.
    ax.axhline(0.5, color=GRAY, lw=0.9, ls="--", zorder=1)
    ax.text(
        len(groups) - 0.5,
        0.512,
        "coin flip",
        fontsize=7,
        color=GRAY,
        ha="right",
        va="bottom",
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(groups.keys(), fontsize=7)
    ax.set_ylabel("AUC, upper vs lower", fontsize=8, color=TEXT)
    ax.set_ylim(0.4, 1.0)
    style_axes(ax)
    ax.legend(fontsize=7, frameon=False, loc="upper left", labelcolor=TEXT)
    ax.set_title(
        "D  Is the cue even in the signal?", fontsize=9, color=TEXT, loc="left", pad=8
    )


def main():
    """Build the error-analysis figure from saved predictions."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--predictions",
        default=os.path.join(RESULTS_DIR, "predictions_official_fold0.npz"),
        help="npz written by `imu2text.models --save-predictions`",
    )
    ap.add_argument(
        "--onhw-chars",
        default=None,
        help="extracted OnHW-chars .npy folder; enables panel D",
    )
    ap.add_argument("--case", default="both")
    ap.add_argument("--dependency", default="indep")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "error_analysis"))
    args = ap.parse_args()

    if not os.path.exists(args.predictions):
        raise SystemExit(
            f"{args.predictions} not found. Produce it with:\n"
            "  python -m imu2text.models --models cnn_bilstm_attn "
            "--onhw-chars data/onhw-chars_2021-06-30 --case both "
            "--dependency indep --fold 0 --epochs 30 --augment 2 "
            "--aug-policy extended --label-smoothing 0.1 --lr-schedule "
            f"--save-predictions {args.predictions}"
        )

    data = np.load(args.predictions, allow_pickle=True)
    true, pred = np.asarray(data["true"]), np.asarray(data["pred"])
    classes = [str(c) for c in data["classes"]]
    acc = 100 * (true == pred).mean()

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6))
    panel_confusion(axes[0, 0], true, pred, classes)
    panel_top_confusions(axes[0, 1], true, pred, classes)
    panel_per_letter(axes[1, 0], true, pred, classes)
    if args.onhw_chars:
        panel_separability(
            axes[1, 1], args.onhw_chars, args.case, args.dependency, args.fold
        )
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5,
            0.5,
            "panel D needs --onhw-chars",
            ha="center",
            va="center",
            fontsize=8,
            color=MUTED,
        )

    model = str(data["model"]) if "model" in data else "model"
    fig.suptitle(
        f"OnHW-chars {len(classes)}-class, writer-independent "
        f"({args.case}/{args.dependency}/fold{args.fold}): "
        f"{acc:.1f}% test accuracy, and where the other {100 - acc:.1f}% goes",
        fontsize=10.5,
        color=TEXT,
        x=0.012,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.012,
        0.012,
        f"{model}, {len(true):,} test samples. Panel A drops the diagonal: only "
        "errors are drawn.",
        fontsize=7,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.028, 1, 0.965))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        path = f"{args.out}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")

    # The table view: every number in the figure, reachable without colour.
    csv_path = f"{args.out}_confusions.csv"
    wrong = true != pred
    pairs = {}
    for t, p in zip(true[wrong], pred[wrong]):
        pairs[(int(t), int(p))] = pairs.get((int(t), int(p)), 0) + 1
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("true,predicted,count,share_of_errors,is_case_pair\n")
        for (t, p), count in sorted(pairs.items(), key=lambda kv: -kv[1]):
            same = classes[t].lower() == classes[p].lower()
            f.write(
                f"{classes[t]},{classes[p]},{count},"
                f"{count / wrong.sum():.4f},{int(same)}\n"
            )
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
