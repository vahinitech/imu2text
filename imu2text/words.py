"""OnHW-words500 dataset loader + lexicon-constrained CTC decoding.

OnHW-words500 is the closed-vocabulary sequence-to-sequence OnHW dataset:
~50 writers each wrote the *same* 500 German words, for a total of 25,218
samples. The 500-word vocabulary is the key lever - any decent CTC model
produces a posterior distribution over characters, and constraining the
final decode to the 500-word lexicon rules out every output that is not one
of the 500 words. How much that is worth on this data has not been measured
here - run it before quoting a WER.

This module provides:

1. ``load_onhw_words500`` - load the per-fold train/val pickles that ship
   with the OnHW-Words500 download. Returns a unified
   ``OnHWWordsDataset`` with the IMU sequences, the integer-encoded label
   sequences, the writer IDs, and the decoded string labels.

2. ``WORDS500_VOCAB`` - the canonical 59-character charset
   (A-Z + a-z + German umlauts ÄÖÜäöüß).

3. ``LexiconDecoder`` - lexicon-constrained beam search over the CTC
   posteriors. For each beam, the decoder tracks the in-progress character
   sequence; only beams whose prefix is a prefix of some lexicon word are
   kept. At the final step, beams that match a lexicon word exactly are
   given a bonus. This is the standard closed-vocabulary HWR decoding
   recipe.

4. A ``--demo`` mode that synthesizes a tiny 50-word lexicon and verifies
   compares lexicon-constrained and greedy decoding on synthetic CTC
   posteriors noisy enough that greedy makes mistakes. It demonstrates the
   mechanism; it is not a measurement on real data.

File layout of OnHW-Words500
----------------------------
Each ``OnHW-Words500_*.zip`` archive extracts to a folder with this
structure (one subfolder per fold, each containing train and val splits):

    Words500_indep_L/                  # or Words500_dep_L, _dep, _indep
    ├── 0/                             # fold directories are bare integers
    │   ├── all_x_dat_train_imu.pkl    # list[np.ndarray (T, 13)]
    │   ├── all_x_dat_val_imu.pkl
    │   ├── all_train_gt.pkl           # list[list[int]], padded to a fixed width
    │   ├── all_val_gt.pkl
    │   ├── train_ids.pkl              # list[int] writer IDs
    │   ├── val_ids.pkl
    │   ├── train_cals.pkl             # per-sample sensor calibration (unused here)
    │   └── val_cals.pkl
    ├── 1/
    └── ids_info.txt                   # which writer IDs are in which fold

Verified against ``OnHW-Words500_indep_L.zip`` (2022-05-20): the fold
directories are named ``0``/``1``, not ``fold_0``. ``load_onhw_words500``
accepts both spellings so it also works with a manually renamed extract.

Label sequences are **right-padded to a fixed width** (19 for the L
archives) with the blank index ``WORDS500_BLANK_IDX`` (= ``len(vocab)`` =
59). ``load_onhw_words500`` strips that padding, so ``Y_train`` holds the
true variable-length token sequences that CTC expects.

The integer-encoded labels use the 59-character charset:

    ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜäöüß
    0:25                                26:51                  52:56

Usage
-----
    from imu2text.words import load_onhw_words500, WORDS500_VOCAB

    ds = load_onhw_words500("./data/Words500_dep_R", fold=0)
    X_train, Y_train = ds.X_train, ds.Y_train
    lexicon = list(set(ds.train_words + ds.val_words))   # 500 unique words

    # train a CTC model (use onhw_seq2seq.build_ctc_models), then decode:
    from imu2text.words import LexiconDecoder
    decoder = LexiconDecoder(lexicon, charset=WORDS500_VOCAB)
    hyps = decoder.decode(infer_model, X_test, down_len)
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, NamedTuple, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Charset and vocabulary
# --------------------------------------------------------------------------- #
# OnHW-words500 uses a 59-character German charset: A-Z (26), a-z (26), and
# the seven German umlauts ÄÖÜäöüß (7). The integer-encoded .pkl labels use
# this exact order. (Earlier docs referenced "57" - that count omitted the
# lowercase ä ö ü ß which are distinct from their uppercase counterparts.)
WORDS500_VOCAB = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜäöüß"
WORDS500_BLANK_IDX = len(WORDS500_VOCAB)  # CTC blank = index 59


def _vocab_index_table() -> Dict[str, int]:
    return {ch: i for i, ch in enumerate(WORDS500_VOCAB)}


def encode_word(word: str) -> List[int]:
    """Encode a string into OnHW-words500 integer tokens (0..56)."""
    table = _vocab_index_table()
    return [table[ch] for ch in word]


def decode_tokens(tokens: Sequence[int]) -> str:
    """Decode integer tokens (0..56) back to a string. Drops -1 / blank."""
    return "".join(WORDS500_VOCAB[t] for t in tokens if 0 <= t < len(WORDS500_VOCAB))


# --------------------------------------------------------------------------- #
# Dataset container
# --------------------------------------------------------------------------- #
class OnHWWordsDataset(NamedTuple):
    """Unified container for an OnHW-words500 fold.

    Attributes
    ----------
    X_train, X_val : list[np.ndarray]
        IMU sequences, each (T, 13) float.
    Y_train, Y_val : list[list[int]]
        Per-sample label sequences (token indices in 0..56).
    train_words, val_words : list[str]
        Decoded string labels (one per sample).
    train_ids, val_ids : np.ndarray
        Per-sample writer IDs.
    fold : int
        Fold index (0-4) this data came from.
    format : str
        Always "words500_pkl".
    """

    X_train: List[np.ndarray]
    X_val: List[np.ndarray]
    Y_train: List[List[int]]
    Y_val: List[List[int]]
    train_words: List[str]
    val_words: List[str]
    train_ids: np.ndarray
    val_ids: np.ndarray
    fold: int
    format: str = "words500_pkl"

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_val(self) -> int:
        return len(self.X_val)

    @property
    def n_writers(self) -> int:
        return len(set(self.train_ids.tolist() + self.val_ids.tolist()))

    @property
    def vocab_size(self) -> int:
        return len(WORDS500_VOCAB)

    @property
    def lexicon(self) -> List[str]:
        """Sorted list of unique words across train+val (the closed 500-word vocab)."""
        return sorted(set(self.train_words + self.val_words))

    def summary(self) -> str:
        lens = [len(s) for s in self.X_train] + [len(s) for s in self.X_val]
        return (
            f"OnHW-words500 (fold {self.fold}): "
            f"train={self.n_train} val={self.n_val} "
            f"writers={self.n_writers} "
            f"lexicon={len(self.lexicon)} words "
            f"len mean={np.mean(lens):.0f} max={max(lens)}"
        )


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def _find_fold_dir(base_dir: str, fold: int) -> str:
    """Resolve the directory holding one fold.

    The shipped archives name their fold directories with bare integers
    (``0``, ``1``, ...). Older documentation and some manual extracts use
    ``fold_0``; we accept either so a renamed directory still loads.
    """
    for name in (str(fold), f"fold_{fold}"):
        candidate = os.path.join(base_dir, name)
        if os.path.isdir(candidate):
            return candidate
    available = sorted(
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    )
    raise FileNotFoundError(
        f"fold {fold} not found under {base_dir} (looked for '{fold}' and "
        f"'fold_{fold}'). Subdirectories present: {available or 'none'}. "
        "Did you download and extract onhw_words500_dep / onhw_words500_indep?"
    )


def _strip_padding(seq: Sequence[int]) -> List[int]:
    """Drop the trailing blank padding from one shipped label sequence.

    The archives right-pad every label to a fixed width with the blank
    index. CTC needs the true length, and a target that is longer than the
    model's output would make the loss undefined, so the padding has to go
    before the sequence is used as a target.
    """
    out = [int(t) for t in seq]
    while out and out[-1] == WORDS500_BLANK_IDX:
        out.pop()
    return out


def load_onhw_words500(
    base_dir: str, fold: int = 0, min_len: int = 1
) -> OnHWWordsDataset:
    """Load one fold of the OnHW-words500 dataset.

    Parameters
    ----------
    base_dir : str
        Path to the extracted Words500 folder. The loader expects a fold
        subfolder - named ``0``..``4`` in the shipped archives, or
        ``fold_0``..``fold_4`` if you renamed them - containing the six
        standard .pkl files: ``all_x_dat_{train,val}_imu.pkl``,
        ``all_{train,val}_gt.pkl``, ``{train,val}_ids.pkl``.
    fold : int, default 0
        Fold index (0-4).
    """
    if not 0 <= fold <= 4:
        raise ValueError(f"fold must be 0-4, got {fold}")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"directory not found: {base_dir}")

    fold_dir = _find_fold_dir(base_dir, fold)

    required = [
        "all_x_dat_train_imu.pkl",
        "all_x_dat_val_imu.pkl",
        "all_train_gt.pkl",
        "all_val_gt.pkl",
        "train_ids.pkl",
        "val_ids.pkl",
    ]
    for fname in required:
        if not os.path.exists(os.path.join(fold_dir, fname)):
            raise FileNotFoundError(
                f"missing {fname} in {fold_dir}. The Words500 archive may be "
                "corrupted; re-download with "
                "`python -m imu2text.download onhw_words500_dep`"
            )

    def _load_pkl(name):
        with open(os.path.join(fold_dir, name), "rb") as f:
            return pickle.load(f)

    X_train = [
        np.asarray(s, dtype=np.float32) for s in _load_pkl("all_x_dat_train_imu.pkl")
    ]
    X_val = [
        np.asarray(s, dtype=np.float32) for s in _load_pkl("all_x_dat_val_imu.pkl")
    ]
    Y_train = [_strip_padding(lab) for lab in _load_pkl("all_train_gt.pkl")]
    Y_val = [_strip_padding(lab) for lab in _load_pkl("all_val_gt.pkl")]
    train_ids = np.array(list(_load_pkl("train_ids.pkl")), dtype=np.int64)
    val_ids = np.array(list(_load_pkl("val_ids.pkl")), dtype=np.int64)

    # Decode token sequences to strings for the lexicon and human-readable eval.
    train_words = [decode_tokens(seq) for seq in Y_train]
    val_words = [decode_tokens(seq) for seq in Y_val]

    # Sanity: token indices must be within the charset, or equal to the
    # padding/blank index (the shipped labels are padded to a fixed width).
    all_tokens = [t for seq in (Y_train + Y_val) for t in seq]
    if all_tokens:
        max_tok = max(all_tokens)
        if max_tok > WORDS500_BLANK_IDX:
            raise ValueError(
                f"found token {max_tok} but the words500 charset only has "
                f"{len(WORDS500_VOCAB)} symbols (+ blank at {WORDS500_BLANK_IDX}) "
                "- the data may be encoded with a different charset"
            )

    # A handful of recordings in the published archives have zero timesteps
    # (3 of 19,918 train and 8 of 5,300 val in indep fold 0). They carry no
    # signal and cannot be normalized, so they are dropped - loudly, and with
    # per-split counts, because dropping val samples changes the denominator
    # of any error rate computed from them.
    keep_tr = [i for i, s_ in enumerate(X_train) if len(s_) >= min_len]
    keep_va = [i for i, s_ in enumerate(X_val) if len(s_) >= min_len]
    if len(keep_tr) < len(X_train) or len(keep_va) < len(X_val):
        print(
            f"  dropped recordings shorter than {min_len} timestep(s): "
            f"{len(X_train) - len(keep_tr)} of {len(X_train)} train, "
            f"{len(X_val) - len(keep_va)} of {len(X_val)} val"
        )
        X_train = [X_train[i] for i in keep_tr]
        Y_train = [Y_train[i] for i in keep_tr]
        train_words = [train_words[i] for i in keep_tr]
        train_ids = train_ids[keep_tr]
        X_val = [X_val[i] for i in keep_va]
        Y_val = [Y_val[i] for i in keep_va]
        val_words = [val_words[i] for i in keep_va]
        val_ids = val_ids[keep_va]

    return OnHWWordsDataset(
        X_train=X_train,
        X_val=X_val,
        Y_train=Y_train,
        Y_val=Y_val,
        train_words=train_words,
        val_words=val_words,
        train_ids=train_ids,
        val_ids=val_ids,
        fold=fold,
    )


# --------------------------------------------------------------------------- #
# Lexicon-constrained beam-search decoder
# --------------------------------------------------------------------------- #
def _accumulate(
    beams: Dict[str, List[float]], prefix: str, slot: int, value: float
) -> None:
    """Add ``value`` to one half of a beam's log-probability pair.

    ``slot`` 0 holds the mass of alignments ending in a blank and slot 1 the
    mass of those ending in the last emitted label. Values are summed in log
    space, since a prefix is usually reachable by several alignments and CTC
    scores the sequence, not its best path.
    """
    entry = beams.get(prefix)
    if entry is None:
        entry = [-np.inf, -np.inf]
        beams[prefix] = entry
    entry[slot] = np.logaddexp(entry[slot], value)


class LexiconDecoder:
    """Lexicon-constrained beam-search CTC decoder for closed-vocabulary HWR.

    Standard CTC greedy decoding collapses repeated symbols and drops blanks,
    but does not use any language-model prior. When the target vocabulary is
    closed (like OnHW-words500's 500 words), we can prune the beam at every
    step to prefixes that are still a prefix of some lexicon word - this
    eliminates most garbage outputs at near-zero cost.

    The decoder is intentionally simple (no LM, no word-boundary handling)
    so it stays a drop-in replacement for ``imu2text.seq2seq.ctc_greedy_decode``.
    For real production use, a proper beam search with a character-level LM
    (e.g. KenLM) is recommended; this implementation gets the closed-vocab
    bonus for free.

    Parameters
    ----------
    lexicon : list[str]
        The closed vocabulary (e.g. the 500 unique words of OnHW-words500).
    charset : str, default WORDS500_VOCAB
        The 59-character charset. Token indices in the CTC output are
        expected to be 0..len(charset)-1, with the CTC blank at
        len(charset).
    beam_width : int, default 8
        Maximum number of beams to keep at each timestep. Higher = more
        accurate but slower.
    lexicon_bonus : float, default 1.0
        Log-probability bonus added to beams whose final decode matches a
        lexicon word exactly. Set to 0 to disable the final rescoring.
    strict : bool, default True
        When no beam spells a complete lexicon word, return "" (an explicit
        no-decode) rather than a partial prefix that is certainly wrong. Set
        False to return the best prefix instead, for partial CER credit.
    """

    def __init__(
        self,
        lexicon: Sequence[str],
        charset: str = WORDS500_VOCAB,
        beam_width: int = 8,
        lexicon_bonus: float = 1.0,
        strict: bool = True,
    ):
        self.charset = charset
        self.beam_width = beam_width
        self.lexicon_bonus = lexicon_bonus
        self.strict = strict
        # Prefix set: every proper prefix of every lexicon word, including "".
        # A beam whose running prefix is absent from this set can never grow
        # into a lexicon word, so it is pruned immediately.
        self._prefixes: set = set()
        for word in lexicon:
            for i in range(len(word) + 1):
                self._prefixes.add(word[:i])
        self._full_words = set(lexicon)

    def _is_valid_prefix(self, prefix: str) -> bool:
        """True if `prefix` is a prefix of some lexicon word."""
        return prefix in self._prefixes

    def _is_full_word(self, word: str) -> bool:
        return word in self._full_words

    def decode_one(self, posteriors: np.ndarray) -> str:
        """Decode one (T, V) posterior matrix to a string.

        ``V`` must equal ``len(charset) + 1`` (the +1 is the CTC blank at the
        last index).

        This is CTC prefix beam search: each beam carries the probability of
        the alignments that end in a blank and those that end in the last
        emitted label, kept separately, and the two are *summed* over
        alignments rather than maximised. Tracking them apart is what makes
        doubled letters decodable at all - "ALL" is only reachable when the
        second L extends a blank-ending alignment - and summing is what makes
        the score an actual probability of the label sequence rather than of
        its single best alignment.

        With ``strict=True`` (the default) the result is always either a
        lexicon word or the empty string: the task is closed-vocabulary, so a
        partial prefix like "BE" is never a legitimate answer, and an empty
        decode reports the miss. Pass ``strict=False`` to fall back
        to the best prefix instead, which scores better under CER.
        """
        T, V = posteriors.shape
        blank = len(self.charset)
        if V != blank + 1:
            raise ValueError(
                f"posteriors have {V} columns but charset has {blank} symbols; "
                f"expected {blank + 1} (charset + CTC blank)"
            )
        log_p = np.log(np.asarray(posteriors, dtype=np.float64) + 1e-12)

        NEG = -np.inf
        # prefix -> [log P(alignments ending in blank),
        #            log P(alignments ending in the last emitted label)]
        beams: Dict[str, List[float]] = {"": [0.0, NEG]}

        for t in range(T):
            row = log_p[t]
            nxt: Dict[str, List[float]] = {}

            for prefix, (pb, pnb) in beams.items():
                total = np.logaddexp(pb, pnb)
                # Blank: the label sequence is unchanged, alignment ends blank.
                _accumulate(nxt, prefix, 0, total + row[blank])
                last = prefix[-1] if prefix else None
                for tok in range(blank):
                    ch = self.charset[tok]
                    p = row[tok]
                    if ch == last:
                        # Repeating the last label collapses into it, so the
                        # sequence is unchanged and the alignment ends on it.
                        _accumulate(nxt, prefix, 1, pnb + p)
                        # Emitting a genuine second copy needs a blank first,
                        # so only the blank-ending mass can extend the prefix.
                        grown = prefix + ch
                        if self._is_valid_prefix(grown):
                            _accumulate(nxt, grown, 1, pb + p)
                    else:
                        grown = prefix + ch
                        if self._is_valid_prefix(grown):
                            _accumulate(nxt, grown, 1, total + p)

            if not nxt:  # nothing survived pruning
                break
            beams = dict(
                sorted(nxt.items(), key=lambda kv: -np.logaddexp(kv[1][0], kv[1][1]))[
                    : self.beam_width
                ]
            )

        scored = [(np.logaddexp(pb, pnb), s) for s, (pb, pnb) in beams.items()]
        words = [
            (lp + self.lexicon_bonus, s) for lp, s in scored if self._is_full_word(s)
        ]
        if words:
            return max(words)[1]
        # No beam spelled a complete word. On a closed vocabulary a partial
        # prefix is a guaranteed error, so the default is to say so with an
        # empty decode rather than emit something that cannot be right.
        # strict=False returns the best prefix instead, which is worth having
        # when the metric is CER and partial credit counts.
        if self.strict:
            return ""
        return max(scored)[1] if scored else ""

    def decode(
        self, infer_model, X: np.ndarray, down_len: int, batch: int = 32
    ) -> List[str]:
        """Run lexicon-constrained beam search on a batch of IMU sequences.

        Parameters
        ----------
        infer_model : keras.Model
            The CTC inference model (input IMU -> output per-frame softmax).
        X : np.ndarray
            Padded IMU input tensor, shape (N, T, C).
        down_len : int
            Number of frames in the model's output (after CNN downsampling).
            Must match what the model was trained with.
        batch : int, default 32
            Mini-batch size for the forward pass.
        """
        out: List[str] = []
        for i in range(0, len(X), batch):
            chunk = X[i : i + batch]
            preds = infer_model.predict(chunk, verbose=0)
            for j in range(len(chunk)):
                out.append(self.decode_one(preds[j][:down_len]))
        return out


# --------------------------------------------------------------------------- #
# Demo / smoke test (no real data needed)
# --------------------------------------------------------------------------- #
def _demo():
    """Synthetic lexicon demo: compare greedy and lexicon-constrained decoding.

    The posteriors are deliberately noisy - each character frame keeps only a
    fraction of its mass on the right symbol and spreads the rest over
    plausible confusions. Clean posteriors would let greedy decoding score
    100% too, which demonstrates nothing; the closed vocabulary only earns
    its keep when the model is unsure.
    """
    rng = np.random.default_rng(0)
    lexicon = [
        "HALLO",
        "WELT",
        "PYTHON",
        "CTC",
        "BEAM",
        "DECODE",
        "LEXICON",
        "WORD",
        "TEST",
        "BERLIN",
    ]
    charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    decoder = LexiconDecoder(lexicon, charset=charset, beam_width=4)

    # Build a synthetic CTC posterior: each word emits its chars at distinct
    # timesteps, with some noise + a few blank frames in between.
    def make_posteriors(word, T=40, confidence=0.18):
        n_chars = len(charset)
        blank = n_chars
        post = np.full((T, n_chars + 1), 0.05, dtype=np.float64)
        post[:, blank] = 0.5
        # Place each character of `word` at evenly-spaced frames, keeping only
        # `confidence` of the mass on the correct symbol and scattering the
        # rest so greedy decoding makes mistakes a lexicon can repair.
        for i, ch in enumerate(word):
            t = int((i + 0.5) * T / len(word))
            tok = charset.index(ch)
            post[t, :] = rng.uniform(0.01, 0.30, size=n_chars + 1)
            post[t, tok] = confidence
            post[t, blank] = 0.05
        post /= post.sum(axis=1, keepdims=True)
        return post

    correct_greedy = 0
    correct_lexicon = 0
    for word in lexicon * 3:
        post = make_posteriors(word)
        # Greedy: argmax per frame, collapse repeats, drop blanks
        greedy_tokens = []
        prev = -1
        for t in range(post.shape[0]):
            tok = int(np.argmax(post[t]))
            if tok != prev and tok != len(charset):
                greedy_tokens.append(charset[tok])
            prev = tok
        greedy_word = "".join(greedy_tokens)
        lexicon_word = decoder.decode_one(post)
        if greedy_word == word:
            correct_greedy += 1
        if lexicon_word == word:
            correct_lexicon += 1
    n = len(lexicon) * 3
    print(f"Greedy decode:  {correct_greedy}/{n} correct ({correct_greedy/n*100:.0f}%)")
    print(
        f"Lexicon decode: {correct_lexicon}/{n} correct ({correct_lexicon/n*100:.0f}%)"
    )
    print(f"Lexicon: {lexicon}")


def main() -> None:
    """CLI: run the lexicon-decode demo, or summarise a Words500 fold."""
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base_dir", nargs="?", help="path to extracted Words500 folder")
    ap.add_argument("--fold", type=int, default=0, help="fold index 0-4 (default 0)")
    ap.add_argument(
        "--demo",
        action="store_true",
        help="run the synthetic lexicon-decode demo (no data needed)",
    )
    args = ap.parse_args()

    if args.demo:
        _demo()
        return
    if not args.base_dir:
        ap.error("pass a base_dir, or use --demo")

    ds = load_onhw_words500(args.base_dir, fold=args.fold)
    print(ds.summary())
    print(f"First 5 train words: {ds.train_words[:5]}")
    print(f"First 5 val words:   {ds.val_words[:5]}")
    print(f"Lexicon size: {len(ds.lexicon)} unique words")
    print(f"First 10 lexicon words: {ds.lexicon[:10]}")


if __name__ == "__main__":
    main()
