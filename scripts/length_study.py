"""What ``--max-len`` cuts from OnHW-chars recordings (issue #15), no training.

``imu2text.models`` pads and truncates at the end (``truncating="post"``),
so a recording longer than ``--max-len`` keeps its first steps and loses the
rest. For one official split this reports, for several cut lengths: how many
recordings are cut, split by capitals and small letters and by train and
test, which letters are cut most, and how many cut recordings lose all of
their writing, because the pen touched the paper only after the cut.

"Writing" is pen-tip force (channel 12) above a threshold: by default the
median force over the split's recordings, a heuristic for pen on paper, not
a calibrated contact detector (the archive carries no sensor calibration).

    python -m scripts.length_study data/onhw-chars_2021-06-30
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

FORCE = 12
CUTS = (72, 100, 128, 160)


def cut_summary(lengths, labels, is_test, cut: int) -> dict:
    """How many recordings a cut at ``cut`` steps truncates, and whose."""
    lengths = np.asarray(lengths)
    labels = np.asarray(labels, dtype=str)
    is_test = np.asarray(is_test, bool)
    upper = np.char.isupper(labels)
    cut_mask = lengths > cut

    def share(mask):
        return float(cut_mask[mask].mean() * 100) if mask.any() else 0.0

    per_class = Counter(labels[cut_mask].tolist())
    totals = Counter(labels.tolist())
    return {
        "cut": cut,
        "truncated": int(cut_mask.sum()),
        "share": share(np.ones(len(lengths), bool)),
        "share_upper": share(upper),
        "share_lower": share(~upper),
        "upper_share_of_truncated": (
            float(upper[cut_mask].mean() * 100) if cut_mask.any() else 0.0
        ),
        "truncated_test": int((cut_mask & is_test).sum()),
        "steps_lost_share": float(
            (lengths[cut_mask] - cut).sum() / max(lengths.sum(), 1) * 100
        ),
        "top_classes": [
            (c, n, round(n / totals[c] * 100, 1)) for c, n in per_class.most_common(10)
        ],
    }


def writing_lost(recordings, is_test, cut: int, threshold: float) -> dict:
    """Of the recordings a cut truncates, how many lose all their writing."""
    lost = lost_test = 0
    kept = []
    truncated = 0
    for rec, test in zip(recordings, is_test):
        if len(rec) <= cut:
            continue
        truncated += 1
        down = np.asarray(rec)[:, FORCE] > threshold
        if not down.any():
            continue
        inside = down[:cut].sum() / down.sum()
        if inside == 0:
            lost += 1
            lost_test += int(test)
        else:
            kept.append(inside)
    return {
        "cut": cut,
        "truncated": truncated,
        "all_writing_lost": lost,
        "all_writing_lost_test": lost_test,
        "median_writing_kept": float(np.median(kept) * 100) if kept else 0.0,
    }


def main() -> None:
    """CLI: print the study for one official split."""
    from imu2text.chars import _load_npy_split  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("data", help="extracted onhw-chars_2021-06-30 folder")
    ap.add_argument("--case", default="both")
    ap.add_argument("--dependency", default="indep")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cuts", type=int, nargs="+", default=list(CUTS))
    ap.add_argument(
        "--force-threshold",
        type=float,
        help="pen-down force; default: median force over the split",
    )
    args = ap.parse_args()

    x_tr, y_tr, x_te, y_te, classes = _load_npy_split(
        args.data, args.case, args.dependency, args.fold
    )
    pairs = [
        (np.asarray(r, np.float64), classes[y], test)
        for xs, ys, test in ((x_tr, y_tr, False), (x_te, y_te, True))
        for r, y in zip(xs, ys)
        if len(r)
    ]
    recs = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]
    is_test = [p[2] for p in pairs]
    lengths = np.array([len(r) for r in recs])
    threshold = args.force_threshold
    if threshold is None:
        threshold = float(np.median(np.concatenate([r[:, FORCE] for r in recs])))

    upper = np.char.isupper(np.asarray(labels, dtype=str))
    print(
        f"{args.case}/{args.dependency}/fold{args.fold}: {len(recs)} recordings, "
        f"length mean {lengths.mean():.1f}, median {np.median(lengths):.0f}, "
        f"p90 {np.percentile(lengths, 90):.0f}, p99 {np.percentile(lengths, 99):.0f}, "
        f"max {lengths.max()}; capitals {lengths[upper].mean():.1f} against "
        f"small letters {lengths[~upper].mean():.1f} steps on average. "
        f"Pen-down force threshold {threshold:.0f}."
    )
    for cut in args.cuts:
        s = cut_summary(lengths, labels, is_test, cut)
        w = writing_lost(recs, is_test, cut, threshold)
        print(
            f"max-len {cut}: {s['truncated']} cut ({s['share']:.2f}%: capitals "
            f"{s['share_upper']:.2f}%, small {s['share_lower']:.2f}%; capitals are "
            f"{s['upper_share_of_truncated']:.1f}% of those cut), {s['truncated_test']} "
            f"in test, {s['steps_lost_share']:.1f}% of all steps dropped. All writing "
            f"after the cut: {w['all_writing_lost']} ({w['all_writing_lost_test']} in "
            f"test); the rest keep a median {w['median_writing_kept']:.0f}% of their "
            f"pen-down steps. Most cut: "
            + ", ".join(f"{c} {n} ({p}%)" for c, n, p in s["top_classes"][:6])
        )
    order = np.argsort(-lengths)[:5]
    for i in order:
        down = np.flatnonzero(recs[i][:, FORCE] > threshold)
        span = f"steps {down[0]} to {down[-1]}" if len(down) else "none"
        print(
            f"  longest: {lengths[i]} steps, '{labels[i]}', "
            f"{'test' if is_test[i] else 'train'}, pen down {len(down)} steps ({span})"
        )


if __name__ == "__main__":
    main()
