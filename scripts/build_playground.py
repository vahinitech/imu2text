"""Export the data the playground page shows.

The page (``playground/index.html``) is plain HTML and JavaScript with no
server and no ML in the browser. Everything it shows comes from this script,
so it can only ever show what the Python code produced.

Two outputs:

``playground/data/public.js`` (committed)
    Model outputs for a selection of official OnHW-chars test letters: each
    ensemble member's softmax and the ensemble mean. No recordings. Also one
    synthetic signal, run through the real filters, so the filter step has
    something to draw.

``playground/data/local.js`` (gitignored)
    The same letters plus their recorded signals, raw and filtered. Written
    only with ``--onhw-chars``, from data you downloaded yourself. The OnHW
    recordings belong to Fraunhofer IIS, are for non-commercial use, and are
    not redistributed by this repo.

    python -m scripts.build_playground --members 'results/ensemble/right_seed*.npz'
    python -m scripts.build_playground --members 'results/ensemble/right_seed*.npz' \\
        --onhw-chars data/onhw-chars_2021-06-30
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from imu2text.chars import CHANNEL_NAMES
from imu2text.filters import FILTERS, SAMPLE_RATE_HZ
from scripts.ensemble_chars import load_members

OUT_DIR = os.path.join("playground", "data")
PER_KIND = 6
EPS = 1e-12


def entropy_bits(p: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits along the last axis."""
    return -(p * np.log2(p + EPS)).sum(-1)


def pick_letters(members: list, seed: int = 0, hand=(0, -1)) -> list:
    """Choose test letters that show the three situations worth explaining.

    ``clear``: every member picks the right letter with high confidence.
    ``case``: members agree, but the ensemble is split between a letter and
    its other case. ``disagree``: the members' outputs differ the most
    (highest mutual information). ``wrong``: confident and wrong.
    """
    ref = members[0]
    true, classes = ref["true"], ref["classes"].astype(str)
    keep = np.isin(ref["handedness"], hand)  # right-handed by default
    probs = np.stack([m["proba"] for m in members])  # (S, N, C)
    mean = probs.mean(0)
    top = mean.argmax(1)
    votes = probs.argmax(2)  # (S, N)
    unanimous = (votes == votes[0]).all(0)
    mutual_info = entropy_bits(mean) - entropy_bits(probs).mean(0)
    order = np.argsort(-mean, axis=1)
    first, second = order[:, 0], order[:, 1]
    folded = np.char.lower(classes)
    case_pair = (folded[first] == folded[second]) & (first != second)
    margin = mean[np.arange(len(true)), first] - mean[np.arange(len(true)), second]

    rng = np.random.default_rng(seed)
    kinds = {
        "clear": keep & unanimous & (top == true) & (mean.max(1) > 0.9),
        "case": keep
        & case_pair
        & (margin < 0.2)
        & (mutual_info < np.median(mutual_info)),
        "wrong": keep & unanimous & (top != true) & (mean.max(1) > 0.8),
    }

    def distinct_letters(kind, candidates):
        # One recording per letter, so a group shows six different letters.
        seen, out = set(), []
        for i in candidates:
            if true[i] not in seen:
                seen.add(true[i])
                out.append((kind, int(i)))
            if len(out) == PER_KIND:
                break
        return out

    chosen = []
    for kind, mask in kinds.items():
        chosen += distinct_letters(kind, rng.permutation(np.flatnonzero(mask)))
    taken = {i for _, i in chosen}
    by_mi = [i for i in np.argsort(-mutual_info) if keep[i] and i not in taken]
    chosen += distinct_letters("disagree", by_mi)
    return chosen


def rounded(a: np.ndarray, digits: int = 3) -> list:
    """Nested lists with fixed precision, to keep the exported file small."""
    return np.round(np.asarray(a, dtype=np.float64), digits).tolist()


# Rough scales for the synthetic signal, in raw counts, so its converted
# values look like a pen's: gravity on one axis of each accelerometer (16,384
# counts per g, imu2text/filters.py), strokes of a few tenths of a g, rotation
# of tens of degrees per second (14.3 counts per deg/s), a steady field.
SYNTH_OFFSET = [0, 0, 16384, 0, 0, 16384, 0, 0, 0, 2500, -1200, 3800]
SYNTH_AMPLITUDE = [4000, 4000, 2500, 4000, 4000, 2500, 900, 900, 600, 300, 300, 300]


def synthetic_signal(seed: int = 0, length: int = 120) -> np.ndarray:
    """A made-up 13-channel recording: smooth strokes plus sensor noise.

    It exists so the public page can show what each filter does to a signal
    without shipping a real recording. It is not handwriting; only the
    magnitudes are chosen to resemble a pen's.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length) / SAMPLE_RATE_HZ
    out = np.zeros((length, len(CHANNEL_NAMES)), dtype=np.float32)
    for c in range(12):
        f = rng.uniform(1.0, 4.0)
        wave = np.sin(2 * np.pi * f * t + rng.uniform(0, np.pi))
        out[:, c] = SYNTH_OFFSET[c] + SYNTH_AMPLITUDE[c] * wave
        out[:, c] += rng.normal(0, 0.15 * SYNTH_AMPLITUDE[c], length)
    out[:, 12] = np.clip(
        np.sin(np.pi * t / t[-1]) * 900 + rng.normal(0, 40, length), 0, None
    )
    return out


def filtered_versions(seq: np.ndarray) -> dict:
    """The recording through every registered filter, channels first."""
    return {
        name: rounded(fn(seq.astype(np.float32)).T, 1) for name, fn in FILTERS.items()
    }


def letter_records(members, chosen, groups, hand="right") -> list:
    """Model outputs for each chosen letter, per group of ensemble members."""
    ref = members[0]
    classes = ref["classes"].astype(str)
    records = []
    for kind, i in chosen:
        rec = {
            "test_index": i,
            "kind": kind,
            "label": classes[ref["true"][i]],
            "hand": hand,
        }
        for name, group in groups.items():
            probs = np.stack([m["proba"][i] for m in group])
            rec[name] = {
                "seeds": [int(m["seed"]) for m in group],
                "members": rounded(probs),
                "mean": rounded(probs.mean(0)),
            }
        records.append(rec)
    return records


def group_summary(group, hand=(0, -1)) -> dict:
    """Whole-test-set accuracy of each member and of the ensemble."""
    ref = group[0]
    keep = np.isin(ref["handedness"], hand)
    true = ref["true"][keep]
    probs = np.stack([m["proba"][keep] for m in group])
    return {
        "split": str(ref["split"]),
        "model": str(ref["model"]),
        "n_test": int(keep.sum()),
        "member_accuracy": [float((p.argmax(1) == true).mean() * 100) for p in probs],
        "ensemble_accuracy": float((probs.mean(0).argmax(1) == true).mean() * 100),
    }


def word_records(path: str, seed: int = 0) -> dict:
    """Sample words from a saved CTC run, with both decodings and their CER.

    ``right``: greedy decoding already spells the word. ``fixed``: greedy is
    wrong and the lexicon decoder recovers it. ``abstain``: the lexicon
    decoder returns nothing (no lexicon word fits). ``wrong``: both are
    wrong. The CER helper is the one the benchmark used.
    """
    from imu2text.seq2seq import cer  # noqa: PLC0415  (imports TensorFlow)

    with np.load(path, allow_pickle=False) as d:
        refs, greedy, lexicon = d["refs"], d["hyps"], d["lexicon_hyps"]
    kinds = {
        "right": greedy == refs,
        "fixed": (greedy != refs) & (lexicon == refs),
        "abstain": lexicon == "",
        "wrong": (greedy != refs) & (lexicon != refs) & (lexicon != ""),
    }
    rng = np.random.default_rng(seed)
    samples, used = [], set()
    for kind, mask in kinds.items():
        seen = set()
        for i in rng.permutation(np.flatnonzero(mask)):
            # Each word once on the page, so no word sits in two groups.
            if refs[i] in used:
                continue
            used.add(refs[i])
            seen.add(refs[i])
            samples.append(
                {
                    "kind": kind,
                    "ref": str(refs[i]),
                    "greedy": str(greedy[i]),
                    "lexicon": str(lexicon[i]),
                    "cer_greedy": round(100 * cer([refs[i]], [greedy[i]]), 1),
                    "cer_lexicon": round(100 * cer([refs[i]], [lexicon[i]]), 1),
                }
            )
            if len(seen) == PER_KIND:
                break
    return {
        "n": int(len(refs)),
        "cer_greedy": round(100 * cer(list(refs), list(greedy)), 2),
        "cer_lexicon": round(100 * cer(list(refs), list(lexicon)), 2),
        "exact_greedy": round(float((greedy == refs).mean() * 100), 2),
        "exact_lexicon": round(float((lexicon == refs).mean() * 100), 2),
        "empty_lexicon": int((lexicon == "").sum()),
        "source": path.replace(os.sep, "/"),
        "samples": samples,
    }


def task_records(path: str, seed: int = 0) -> dict:
    """Sample test items from one single-model run on another task.

    ``clear``: right with at least 90% confidence. ``unsure``: top choice
    below 50%. ``wrong``: wrong with at least 70% confidence.
    """
    with np.load(path, allow_pickle=False) as d:
        run = {k: d[k] for k in d.files}
    proba, true = run["proba"], run["true"]
    classes = run["classes"].astype(str).tolist()
    top, conf = proba.argmax(1), proba.max(1)
    kinds = {
        "clear": (top == true) & (conf >= 0.9),
        "unsure": conf < 0.5,
        "wrong": (top != true) & (conf >= 0.7),
    }
    rng = np.random.default_rng(seed)
    samples = []
    for kind, mask in kinds.items():
        seen = set()
        for i in rng.permutation(np.flatnonzero(mask)):
            if true[i] in seen:
                continue
            seen.add(true[i])
            samples.append(
                {
                    "test_index": int(i),
                    "kind": kind,
                    "label": classes[true[i]],
                    "probs": rounded(proba[i]),
                }
            )
            if len(seen) == PER_KIND:
                break
    return {
        "classes": classes,
        "split": str(run["split"]),
        "model": str(run["model"]),
        "seed": int(run["seed"]),
        "n_test": int(len(true)),
        "accuracy": round(float((top == true).mean() * 100), 2),
        "source": path.replace(os.sep, "/"),
        "samples": samples,
    }


def write_js(path: str, name: str, payload: dict) -> None:
    """Write ``window.<name> = {...}`` so the page also works from file://."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("// Generated by scripts/build_playground.py. Do not edit.\n")
        f.write(f"window.{name} = ")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")


def main() -> None:
    """CLI: export public.js, and local.js when a data directory is given."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--members", required=True, help="quoted glob, right-handed runs")
    ap.add_argument(
        "--members-left", help="quoted glob, runs trained with left-handed data"
    )
    ap.add_argument(
        "--onhw-chars", help="downloaded right-handed archive, for local.js"
    )
    ap.add_argument(
        "--words",
        default="results/ctc/refit_seed0.json.predictions.npz",
        help="saved Words500 CTC predictions (refs, hyps, lexicon_hyps)",
    )
    ap.add_argument(
        "--task",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="a single-model --save-predictions file for another task, e.g. "
        "symbols_indep=results/tasks/symbols_indep.npz",
    )
    args = ap.parse_args()

    right = load_members(args.members)
    groups = {"right": right}
    if args.members_left:
        both = load_members(args.members_left)
        n = len(right[0]["true"])
        if not np.array_equal(both[0]["true"][:n], right[0]["true"]):
            raise SystemExit("the two groups do not share the right-handed test order")
        groups["with_left"] = both
    chosen = pick_letters(right)
    classes = right[0]["classes"].astype(str).tolist()

    public = {
        "classes": classes,
        "channels": CHANNEL_NAMES,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "groups": {name: group_summary(g) for name, g in groups.items()},
        "letters": letter_records(right, chosen, groups),
        "left_letters": (
            letter_records(
                groups["with_left"],
                pick_letters(groups["with_left"], hand=(1,)),
                {"with_left": groups["with_left"]},
                hand="left",
            )
            if "with_left" in groups
            else []
        ),
        "left_summary": (
            group_summary(groups["with_left"], hand=(1,))
            if "with_left" in groups
            else None
        ),
        "tasks": {
            name: task_records(path)
            for name, path in (item.split("=", 1) for item in args.task)
        },
        "words": word_records(args.words) if os.path.exists(args.words) else None,
        "synthetic_signal": filtered_versions(synthetic_signal()),
    }
    write_js(os.path.join(OUT_DIR, "public.js"), "PLAYGROUND_PUBLIC", public)

    if args.onhw_chars:
        from imu2text.models import load_official_split  # noqa: PLC0415

        seed = int(right[0]["split_seed"])
        x, y, _, (_, _, test) = load_official_split(
            args.onhw_chars, "both", "indep", 0, seed
        )
        local = {}
        for _, i in chosen:
            if classes[int(y[test[i]])] != classes[int(right[0]["true"][i])]:
                raise SystemExit(
                    f"test letter {i} does not match the saved predictions"
                )
            local[str(i)] = filtered_versions(x[test[i]])
        write_js(os.path.join(OUT_DIR, "local.js"), "PLAYGROUND_LOCAL", local)


if __name__ == "__main__":
    main()
