"""Figures: the model architectures, and what transfer learning freezes.

Both figures are drawn by introspecting the real Keras models, not from a
hand-maintained description, so a diagram cannot drift from the code that
produced the numbers. Figure 2 reads each layer's ``trainable`` flag off the
model ``build_transfer_model`` actually returns, so what it labels frozen is
frozen. Change a builder in ``imu2text/models.py`` and the
figure changes with it.

Figure 1, architecture. The layer stack of each model side by side, with the
output shape and parameter count of every layer. It answers what the extra
1.5 points of the attention variant cost: where the parameters sit, and which
stage of the network they are in.

Figure 2, transfer learning. Which layers ``imu2text.symbols
.build_transfer_model`` freezes and which it trains, in the two phases
(head-only warmup, then fine-tune the trunk at a lower learning rate).

Design credit, not code: the layout of Figure 2 follows the shape of Figure 6
in Ott et al., "Domain Adaptation for Time-Series Classification to Mitigate
Covariate Shift" (ACM MM 2022) - frozen block, post-trained block, adaptation
layer read left to right. The drawing code here is written from scratch
against our own models; nothing is taken from theirs.

Usage
-----
    python scripts/plot_architecture.py                 # both figures
    python scripts/plot_architecture.py --figure transfer
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
# pylint: disable=wrong-import-position
# The backend has to be chosen before pyplot is imported.
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pylint: enable=wrong-import-position

# Categorical slots for layer families, in fixed order. Validated with the
# dataviz palette checker: worst adjacent CVD dE 9.6 (deutan), normal-vision
# 20.0. Three slots sit under 3:1 contrast on white, which the checker allows
# only with relief - every block here carries its layer name as visible text,
# and the full table is printed to stdout.
FAMILY = {
    "conv": ("#0072B2", "convolution"),
    "pool": ("#E69F00", "pooling"),
    "norm": ("#009E73", "normalisation"),
    "rnn": ("#D55E00", "recurrent"),
    "dense": ("#56B4E9", "dense"),
    "attn": ("#CC79A7", "attention pooling"),
    "drop": ("#c9c8c4", "dropout (no-op at inference)"),
    "io": ("#f0efec", "input / output"),
}
TEXT = "#0b0b0b"
MUTED = "#83827d"
GRID = "#e6e5e2"
FROZEN = "#8f8e8a"
TRAINED = "#0072B2"

RESULTS_DIR = "results"


def classify(layer):
    """Map a Keras layer onto one of the drawing families."""
    name = type(layer).__name__
    if "Conv" in name:
        return "conv"
    if "Pool" in name and "Global" not in name:
        return "pool"
    if "Normalization" in name:
        return "norm"
    if name in ("LSTM", "GRU", "Bidirectional"):
        return "rnn"
    if name in ("Dense",):
        return "dense"
    if name in ("Softmax", "Dot", "Concatenate") or "Global" in name:
        return "attn"
    if "Dropout" in name:
        return "drop"
    return "io"


def describe(layer):
    """A short label: what the layer is, and the one number that matters."""
    name = type(layer).__name__
    cfg = layer.get_config()
    if "Conv" in name:
        return f"Conv1D {cfg['filters']}"
    if name == "Bidirectional":
        inner = cfg["layer"]["config"]
        return f"BiLSTM {inner['units']}"
    if name in ("LSTM", "GRU"):
        return f"{name} {cfg['units']}"
    if name == "Dense":
        return f"Dense {cfg['units']}"
    if "Pool" in name and "Global" not in name:
        return f"MaxPool {cfg.get('pool_size', ('',))[0]}"
    if "Normalization" in name:
        return "BatchNorm"
    if "Dropout" in name:
        return f"Dropout {cfg['rate']}"
    if name == "InputLayer":
        return "IMU input"
    return {
        "Softmax": "attn weights",
        "Dot": "weighted sum",
        "Concatenate": "concat",
        "Flatten": "flatten",
        "GlobalMaxPooling1D": "max over time",
        "Masking": "mask pad",
    }.get(name, name)


def out_shape(layer):
    """The layer's output shape without the batch axis, as `T x C`."""
    try:
        shape = layer.output_shape
    except AttributeError:  # pragma: no cover - Keras version differences
        return ""
    if isinstance(shape, list):
        shape = shape[0]
    dims = [d for d in shape[1:] if d is not None]
    return "x".join(str(d) for d in dims)


def draw_stack(ax, model, title, subtitle):
    """Draw one model as a left-to-right run of labelled blocks."""
    layers = [l for l in model.layers if type(l).__name__ != "InputLayer"]
    n = len(layers)
    width, gap = 1.0, 0.30
    for i, layer in enumerate(layers):
        fam = classify(layer)
        color, _ = FAMILY[fam]
        x = i * (width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, 0),
                width,
                3.0,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=color,
                edgecolor="none",
                zorder=2,
            )
        )
        ax.text(
            x + width / 2,
            1.5,
            describe(layer),
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.5,
            color="white" if fam not in ("drop", "io") else TEXT,
            zorder=3,
        )
        ax.text(
            x + width / 2,
            -0.22,
            out_shape(layer),
            ha="center",
            va="top",
            fontsize=6,
            color=MUTED,
            family="monospace",
        )
        params = layer.count_params()
        if params:
            ax.text(
                x + width / 2,
                -0.72,
                f"{params / 1000:.0f}k" if params >= 1000 else str(params),
                ha="center",
                va="top",
                fontsize=6,
                color=MUTED,
            )
        if i < n - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + width, 1.5),
                    (x + width + gap, 1.5),
                    arrowstyle="-|>",
                    mutation_scale=7,
                    color=GRID,
                    lw=1.0,
                    zorder=1,
                )
            )
    ax.set_xlim(-0.5, n * (width + gap) + 0.2)
    ax.set_ylim(-1.5, 4.1)
    ax.axis("off")
    ax.text(0, 3.55, title, fontsize=10, color=TEXT, va="bottom", weight="bold")
    ax.text(0, 3.25, subtitle, fontsize=7.5, color=MUTED, va="bottom")
    ax.text(
        -0.45,
        0.35,
        "shape\nparams",
        fontsize=6,
        color=MUTED,
        ha="right",
        va="top",
        family="monospace",
    )


def figure_architecture(out_path, maxlen=100, n_classes=52):
    """Both classification models, side by side, from the real builders."""
    from imu2text import models as M  # noqa: PLC0415

    M.RNN_UNITS, M.RNN_LAYERS = 64, 1
    specs = [
        (
            "cnn_bilstm",
            "CNN+BiLSTM",
            "the OnHW baseline: read out the BiLSTM's final state",
        ),
        (
            "cnn_bilstm_attn",
            "CNN+BiLSTM + attention pooling",
            "keeps every timestep, learns which ones matter",
        ),
    ]
    built = [(M.BUILDERS[key](maxlen, n_classes), t, s) for key, t, s in specs]

    fig, axes = plt.subplots(len(built), 1, figsize=(13.0, 6.4))
    for ax, (model, title, sub) in zip(axes, built):
        draw_stack(ax, model, title, f"{sub}  -  {model.count_params():,} parameters")

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none")
        for c, _ in FAMILY.values()
    ]
    fig.legend(
        handles,
        [n for _, n in FAMILY.values()],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=7.5,
        labelcolor=TEXT,
        handlelength=1.1,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.suptitle(
        f"imu2text classification architectures (input {maxlen}x13, "
        f"{n_classes} classes)",
        fontsize=11,
        color=TEXT,
        x=0.012,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.012,
        0.075,
        "Drawn by introspecting the Keras models in imu2text/models.py, so the "
        "figure cannot drift from the code.",
        fontsize=7,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    _save(fig, out_path)

    print(f"\n{'model':<22}{'layer':<20}{'family':<16}{'shape':<12}{'params':>10}")
    for model, title, _ in built:
        for layer in model.layers:
            if type(layer).__name__ == "InputLayer":
                continue
            print(
                f"{title[:21]:<22}{describe(layer):<20}"
                f"{FAMILY[classify(layer)][1][:15]:<16}"
                f"{out_shape(layer):<12}{layer.count_params():>10,}"
            )


def figure_transfer(out_path, maxlen=100, n_source=52, n_target=15):
    """What build_transfer_model freezes, read off the real model."""
    from imu2text import models as M  # noqa: PLC0415
    from imu2text.symbols import build_transfer_model  # noqa: PLC0415

    M.RNN_UNITS, M.RNN_LAYERS = 64, 1
    pretrained = M.BUILDERS["cnn_bilstm"](maxlen, n_source)
    transfer = build_transfer_model(pretrained, n_classes=n_target)

    layers = [l for l in transfer.layers if type(l).__name__ != "InputLayer"]
    fig, ax = plt.subplots(figsize=(12.6, 4.8))
    width, gap = 1.05, 0.28

    for i, layer in enumerate(layers):
        fam = classify(layer)
        color, _ = FAMILY[fam]
        x = i * (width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, 0),
                width,
                2.6,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=color,
                edgecolor="none",
                zorder=2,
            )
        )
        label = describe(layer)
        if layer is layers[-1]:
            label += "\n(new head)"
        ax.text(
            x + width / 2,
            1.3,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.5,
            color="white" if fam not in ("drop", "io") else TEXT,
            zorder=3,
        )
        ax.text(
            x + width / 2,
            -0.28,
            "frozen" if not layer.trainable else "trains",
            ha="center",
            va="top",
            fontsize=6.5,
            color=FROZEN if not layer.trainable else TRAINED,
        )
        if i < len(layers) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + width, 1.3),
                    (x + width + gap, 1.3),
                    arrowstyle="-|>",
                    mutation_scale=7,
                    color=GRID,
                    lw=1.0,
                    zorder=1,
                )
            )

    n_frozen = sum(1 for l in layers if not l.trainable)
    span_end = n_frozen * (width + gap) - gap / 2
    ax.add_patch(
        Rectangle(
            (-0.16, -0.95),
            span_end + 0.16,
            3.95,
            facecolor="none",
            edgecolor=FROZEN,
            lw=1.1,
            linestyle=(0, (5, 3)),
            zorder=4,
        )
    )
    ax.text(
        span_end / 2,
        3.12,
        f"cloned from the OnHW-chars model, frozen for the warmup "
        f"({n_frozen} layers)",
        fontsize=8,
        color=FROZEN,
        ha="center",
        va="bottom",
    )
    head_x = n_frozen * (width + gap)
    ax.add_patch(
        Rectangle(
            (head_x - 0.12, -0.95),
            width + 0.24,
            3.95,
            facecolor="none",
            edgecolor=TRAINED,
            lw=1.4,
            zorder=4,
        )
    )
    ax.text(
        head_x + width / 2,
        3.12,
        "trained from scratch",
        fontsize=8,
        color=TRAINED,
        ha="center",
        va="bottom",
    )

    trainable = sum(int(w.shape.num_elements()) for w in transfer.trainable_weights)
    ax.text(
        0,
        -1.35,
        f"Phase 1   head only, {trainable:,} of {transfer.count_params():,} "
        f"parameters train        "
        f"Phase 2   unfreeze_trunk(), all of them train at lr 1e-4",
        fontsize=8,
        color=TEXT,
        va="top",
    )
    ax.text(
        0,
        -1.95,
        "The trunk is cloned before reuse. A functional model built on another "
        "model's tensors shares its layer objects, so freezing or fine-tuning\n"
        "here would reach back and change the pretrained chars model in place.",
        fontsize=7,
        color=MUTED,
        va="top",
    )
    ax.set_xlim(-0.6, len(layers) * (width + gap) + 0.2)
    ax.set_ylim(-3.0, 3.9)
    ax.axis("off")
    fig.suptitle(
        f"Transfer learning: OnHW-chars ({n_source} classes) -> "
        f"OnHW-symbols ({n_target} classes)",
        fontsize=11,
        color=TEXT,
        x=0.012,
        ha="left",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_path)


def _save(fig, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        path = f"{out_path}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def main():
    """Draw the architecture and transfer-learning figures."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--figure", choices=("architecture", "transfer", "both"), default="both"
    )
    ap.add_argument("--maxlen", type=int, default=100)
    ap.add_argument("--classes", type=int, default=52)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    args = ap.parse_args()

    if args.figure in ("architecture", "both"):
        figure_architecture(
            os.path.join(args.out_dir, "architecture"), args.maxlen, args.classes
        )
    if args.figure in ("transfer", "both"):
        figure_transfer(os.path.join(args.out_dir, "transfer_learning"))


if __name__ == "__main__":
    main()
