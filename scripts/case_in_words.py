"""How often a word model gets the case right once it has the word (issue #11).

Reads saved CTC predictions (``refs``, ``hyps`` and, when present,
``lexicon_hyps`` from ``imu2text.seq2seq --save-predictions`` or
``scripts.refit_ctc``); no data download.

For each decoding it counts the words read right ignoring case, and of those
the share also right in case, per word and per character. The word list
holds most words in one spelling, so for those, lexicon decoding sets the
case by construction. The test that means something is the words the test
set has in two spellings (``Wie`` and ``wie``), reported separately.

    python -m scripts.case_in_words results/ctc/refit_seed0.json.predictions.npz
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np


def case_in_words(refs, hyps) -> dict:
    """Case accuracy on the words a decoder read right ignoring case."""
    refs = np.asarray(refs, dtype=str)
    hyps = np.asarray(hyps, dtype=str)
    spellings = defaultdict(set)
    for r in refs:
        spellings[r.lower()].add(r)
    twin = np.array([len(spellings[r.lower()]) > 1 for r in refs], bool)
    folded = np.char.lower(hyps) == np.char.lower(refs)
    exact = hyps == refs
    chars = sum(len(r) for r in refs[folded])
    wrong_chars = sum(
        sum(a != b for a, b in zip(r, h)) for r, h in zip(refs[folded], hyps[folded])
    )

    def share(num, den):
        return float(num / den * 100) if den else 0.0

    return {
        "n": int(len(refs)),
        "exact": share(exact.sum(), len(refs)),
        "case_insensitive": share(folded.sum(), len(refs)),
        "case_right_in_read_words": share((exact & folded).sum(), folded.sum()),
        "case_right_chars_in_read_words": share(chars - wrong_chars, chars),
        "twin_words": int(twin.sum()),
        "twin_read": int((folded & twin).sum()),
        "twin_case_right": int((exact & folded & twin).sum()),
        "case_errors_on_twin_words": int((folded & ~exact & twin).sum()),
        "case_errors": int((folded & ~exact).sum()),
    }


def main() -> None:
    """CLI: print the case figures for each decoding in each file."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("predictions", nargs="+", help="saved CTC predictions (.npz)")
    args = ap.parse_args()
    for path in args.predictions:
        with np.load(path, allow_pickle=False) as d:
            runs = [("greedy", d["hyps"])]
            if "lexicon_hyps" in d.files and len(d["lexicon_hyps"]):
                runs.append(("word list", d["lexicon_hyps"]))
            refs = d["refs"]
            for name, hyps in runs:
                r = case_in_words(refs, hyps)
                print(
                    f"{path} [{name}] {r['n']} words: {r['case_insensitive']:.2f}% read "
                    f"ignoring case, {r['exact']:.2f}% exactly; of the read words "
                    f"{r['case_right_in_read_words']:.1f}% right in case "
                    f"({r['case_right_chars_in_read_words']:.2f}% of their letters). "
                    f"Words in two spellings: {r['twin_case_right']}/{r['twin_read']} "
                    f"right in case; {r['case_errors_on_twin_words']} of "
                    f"{r['case_errors']} case errors are on them."
                )


if __name__ == "__main__":
    main()
