"""Score the playground's drawing pad as the page runs it: two AIs in a row.

The pad cannot hand a drawing to the pen AI, which reads a sensor pen's
movement. The drawing reader (``playground/reader.js`` through
``playground/shapes.js``) names the drawing, and the page then shows the pen
AI's saved answer for one real OnHW recording of that character, the class
median (``one_per_class`` in ``scripts/build_playground.py``). This script
measures each step and the chain on drawings by the 20 UJI Pen Characters
test writers, whom the reader never saw:

* single characters, read the way the page reads them (the task narrows the
  answer: digits on Numbers, letters on Letters);
* sequences of 2 or 3 digits and pairs of letters, built from one test
  writer's characters placed side by side, from slightly overlapping to well
  apart;
* the chain: right on the page's recording, and expected on any recording
  (each correct reader answer times the pen AI's measured rate for that
  character on unseen writers).

Needs Node.js and the UJI file (see ``scripts/drawing_reader.py``):

    python -m scripts.drawing_pad_eval data/ujipenchars2/ujipenchars2.txt \\
        --out results/drawing_reader/pipeline.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess

import numpy as np

from scripts.build_playground import one_per_class
from scripts.drawing_reader import parse_uji

PX_PER_MM = 12.0  # a UJI character drawn about the size people draw on the pad
PAD_H = 170  # the drawing area's height on the page, CSS pixels

RUNNER = """
const path = require("path");
const dir = process.argv[1];
globalThis.window = globalThis;
require(path.join(dir, "reader-weights.js"));
require(path.join(dir, "reader.js"));
const S = require(path.join(dir, "shapes.js"));
let input = "";
process.stdin.on("data", (d) => { input += d; });
process.stdin.on("end", () => {
  const out = JSON.parse(input).map((c) => {
    const r = S.recognise(c.strokes, { size: c.size, task: c.task });
    if (!r) return { read: "", split: false };
    if (r.sequence) return { read: r.sequence.map((p) => p.label || "?").join(""), split: true };
    return { read: r.label || "", split: false };
  });
  process.stdout.write(JSON.stringify(out));
});
"""


def read_all(cases: list, playground: str) -> list:
    """Run the page's recogniser on [{strokes, task}] under Node."""
    node = shutil.which("node")
    if node is None:
        raise SystemExit("needs Node.js")
    payload = [
        {
            "size": PAD_H,
            "task": c["task"],
            "strokes": [[{"x": x, "y": y} for x, y in s] for s in c["strokes"]],
        }
        for c in cases
    ]
    done = subprocess.run(
        [node, "-e", RUNNER, os.path.abspath(playground)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout)


def on_pad(strokes: list) -> list:
    """A UJI drawing (millimetres) in pad pixels, left and vertically centred."""
    pts = np.concatenate(strokes) * PX_PER_MM
    lo, span = pts.min(0), np.ptp(pts, 0)
    origin = lo - [40.0, (PAD_H - span[1]) / 2]
    return [(s * PX_PER_MM - origin).tolist() for s in strokes]


def side_by_side(chars: list, rng) -> list:
    """One writer's characters in a row, bottoms aligned, gaps from -8% to
    25% of the tallest character's height."""
    boxes = [np.concatenate(c) for c in chars]
    height = max(float(np.ptp(b[:, 1])) for b in boxes)
    out, x = [], 40.0
    for i, (c, b) in enumerate(zip(chars, boxes)):
        gap = (-0.08 + rng.random() * 0.33) * height
        dx = (x + gap if i else x) - b[:, 0].min()
        dy = 20 + height - b[:, 1].max()
        out.extend((np.asarray(s) + [dx, dy]).tolist() for s in c)
        x = b[:, 0].max() + dx
    return out


def task_of(label: str) -> str:
    """The page's task for a character: Numbers for digits, else Letters."""
    return "symbols" if label.isdigit() else "chars"


def pen_ai(letters_glob: str, symbols_path: str) -> dict:
    """Per character: is the page's recording read right, and the class rate."""
    out = {}
    members = [np.load(f, allow_pickle=False) for f in sorted(glob.glob(letters_glob))]
    keep = np.isin(members[0]["handedness"], (0, -1))
    true = members[0]["true"][keep]
    proba = np.mean([m["proba"] for m in members], axis=0)[keep]
    sym = np.load(symbols_path, allow_pickle=False)
    for task, (t, p, classes) in {
        "chars": (true, proba, members[0]["classes"].astype(str)),
        "symbols": (sym["true"], sym["proba"], sym["classes"].astype(str)),
    }.items():
        top = p.argmax(1)
        page = {
            int(t[i]): bool(top[i] == t[i]) for i in one_per_class(t, p, len(classes))
        }
        out[task] = {
            str(classes[c]): {
                "page_recording_right": page.get(c),
                "rate": float((top[t == c] == c).mean()) if (t == c).any() else None,
                "n": int((t == c).sum()),
            }
            for c in range(len(classes))
        }
    return out


def pct(a: float, n: int) -> float:
    """Share in percent, one decimal."""
    return round(100.0 * a / n, 1) if n else 0.0


def main() -> None:
    """CLI: score the pad and write one JSON file."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("data", help="ujipenchars2.txt")
    ap.add_argument("--out", default="results/drawing_reader/pipeline.json")
    ap.add_argument("--playground", default="playground")
    ap.add_argument("--letters", default="results/ensemble/right_seed*.npz")
    ap.add_argument("--symbols", default="results/tasks/symbols_indep.npz")
    ap.add_argument("--sequences", type=int, default=200, help="per kind")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    test = [r for r in parse_uji(args.data) if r["split"] == "tst"]
    pen = pen_ai(args.letters, args.symbols)

    singles = [
        {"task": task_of(r["label"]), "strokes": on_pad(r["strokes"])} for r in test
    ]
    reads = read_all(singles, args.playground)
    single = {}
    for kind, digit in (("digits", True), ("letters", False)):
        idx = [i for i, r in enumerate(test) if r["label"].isdigit() == digit]
        n = len(idx)
        right = [i for i in idx if reads[i]["read"] == test[i]["label"]]
        task = "symbols" if digit else "chars"
        on_page = sum(
            pen[task].get(test[i]["label"], {}).get("page_recording_right") is True
            for i in right
        )
        expected = sum(pen[task][test[i]["label"]]["rate"] or 0.0 for i in right)
        single[kind] = {
            "n": n,
            "reader_right": pct(len(right), n),
            "reader_right_ignoring_case": pct(
                sum(reads[i]["read"].lower() == test[i]["label"].lower() for i in idx),
                n,
            ),
            "not_sure": sum(not reads[i]["read"] for i in idx),
            "split_in_two": sum(reads[i]["split"] for i in idx),
            "chain_right_page_recording": pct(on_page, n),
            "chain_right_expected_any_recording": pct(expected, n),
        }

    rng = np.random.default_rng(args.seed)
    writers = sorted({r["writer"] for r in test})
    seqs, truths = [], []
    for task, digit, lengths in (("symbols", True, (2, 3)), ("chars", False, (2,))):
        for _ in range(args.sequences):
            writer = writers[rng.integers(len(writers))]
            pool = [
                r
                for r in test
                if r["writer"] == writer and r["label"].isdigit() == digit
            ]
            chars = [pool[rng.integers(len(pool))] for _ in range(rng.choice(lengths))]
            seqs.append(
                {
                    "task": task,
                    "strokes": side_by_side(
                        [
                            [np.asarray(s) * PX_PER_MM for s in c["strokes"]]
                            for c in chars
                        ],
                        rng,
                    ),
                }
            )
            truths.append("".join(c["label"] for c in chars))
    got = read_all(seqs, args.playground)
    sequences = {}
    for kind, task in (("digits_2_or_3", "symbols"), ("letter_pairs", "chars")):
        idx = [i for i, s in enumerate(seqs) if s["task"] == task]
        sequences[kind] = {
            "n": len(idx),
            "exact": pct(sum(got[i]["read"] == truths[i] for i in idx), len(idx)),
            "ignoring_case": pct(
                sum(got[i]["read"].lower() == truths[i].lower() for i in idx), len(idx)
            ),
        }

    result = {
        "data": "UJI Pen Characters v2, published test split, 20 writers",
        "reader": "results/drawing_reader/summary.json (one training run, seed 0)",
        "pen_ai": {
            "letters": f"{args.letters}, 5-seed ensemble, right-handed test writers",
            "symbols": f"{args.symbols}, seed 0",
            "page_recordings_right": {
                task: f"{sum(v['page_recording_right'] is True for v in d.values())}"
                f" of {sum(v['page_recording_right'] is not None for v in d.values())}"
                for task, d in pen.items()
            },
            "per_character": pen,
        },
        "px_per_mm": PX_PER_MM,
        "single": single,
        "sequences": {"seed": args.seed, **sequences},
        "single_false_split": pct(sum(r["split"] for r in reads), len(reads)),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    print(
        json.dumps(
            {k: result[k] for k in ("single", "sequences", "single_false_split")},
            indent=1,
        )
    )
    print("page recordings right:", result["pen_ai"]["page_recordings_right"])


if __name__ == "__main__":
    main()
