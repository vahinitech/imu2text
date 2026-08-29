"""Tests for the OnHW-words500 loader and lexicon-constrained decoder."""

import os
import pickle
import shutil
import tempfile

import numpy as np
import pytest

import onhw_words as W


# --------------------------------------------------------------------------- #
# Charset / vocabulary tests
# --------------------------------------------------------------------------- #
def test_words500_vocab_size_is_59():
    """59 = 26 uppercase + 26 lowercase + 7 German umlauts (ÄÖÜäöüß)."""
    assert len(W.WORDS500_VOCAB) == 59
    assert W.WORDS500_VOCAB[:26] == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert W.WORDS500_VOCAB[26:52] == "abcdefghijklmnopqrstuvwxyz"
    assert W.WORDS500_VOCAB[52:] == "ÄÖÜäöüß"
    assert W.WORDS500_BLANK_IDX == 59


def test_encode_decode_roundtrip():
    """encode_word and decode_tokens are inverse operations."""
    for word in ["HALLO", "Welt", "Straße", "MÜNCHEN", "äöüßÄÖÜ"]:
        encoded = W.encode_word(word)
        decoded = W.decode_tokens(encoded)
        assert decoded == word, f"roundtrip failed: {word} -> {encoded} -> {decoded}"


def test_decode_tokens_drops_invalid_indices():
    """Negative indices and out-of-range indices are dropped silently."""
    # 0 -> A, 1 -> B, 2 -> C; -1 / 99 / blank(=59) are dropped
    assert W.decode_tokens([-1, 0, 99, 1, -99, 2]) == "ABC"
    assert W.decode_tokens([0, 59, 1]) == "AB"  # 59 = blank index, dropped


# --------------------------------------------------------------------------- #
# Lexicon decoder tests
# --------------------------------------------------------------------------- #
def test_lexicon_decoder_prunes_invalid_prefixes():
    """A beam whose running prefix is not a lexicon prefix is pruned."""
    decoder = W.LexiconDecoder(lexicon=["HALLO", "HILFE"], beam_width=4)
    assert decoder._is_valid_prefix("")
    assert decoder._is_valid_prefix("H")
    assert decoder._is_valid_prefix("HA")
    assert decoder._is_valid_prefix("HAL")
    assert decoder._is_valid_prefix("HALLO")
    assert not decoder._is_valid_prefix("X")  # not a prefix of any word
    assert not decoder._is_valid_prefix("HB")  # H yes, HB no


def test_lexicon_decoder_full_word_check():
    decoder = W.LexiconDecoder(lexicon=["HALLO", "HILFE"], beam_width=4)
    assert decoder._is_full_word("HALLO")
    assert decoder._is_full_word("HILFE")
    assert not decoder._is_full_word("HAL")
    assert not decoder._is_full_word("XYZ")


def test_lexicon_decoder_perfect_posteriors_with_blanks():
    """When the CTC posteriors perfectly spell a lexicon word (with the
    standard CTC blank-between-repeats convention), the decoder must return
    that word (no information loss).

    CTC requires a blank between repeated characters to encode them as two
    distinct emissions rather than a single collapsed repeat. So the word
    "HALLO" must be emitted as H _ A _ L _ blank _ L _ O.
    """
    decoder = W.LexiconDecoder(lexicon=["HALLO", "HILFE"], beam_width=8)
    charset = decoder.charset
    blank = len(charset)
    # Frame sequence: H, blank, A, blank, L, blank, L, blank, O
    seq = ["H", "_", "A", "_", "L", "_", "L", "_", "O"]
    T = len(seq)
    V = blank + 1
    post = np.full((T, V), 1e-6, dtype=np.float32)
    for t, ch in enumerate(seq):
        if ch == "_":
            post[t, blank] = 1.0 - 1e-6 * (V - 1)
        else:
            post[t, charset.index(ch)] = 1.0 - 1e-6 * (V - 1)
    post /= post.sum(axis=1, keepdims=True)
    result = decoder.decode_one(post)
    assert result == "HALLO"


def test_lexicon_decoder_with_blanks_between_chars():
    """CTC requires blanks between repeated characters; the decoder must
    handle the standard CTC collapse-repeats logic correctly."""
    decoder = W.LexiconDecoder(lexicon=["HALLO"], beam_width=8)
    # Posterior: H _ A _ L _ L _ O (blanks between each char)
    charset = decoder.charset
    blank = len(charset)
    sequence = ["H", "_", "A", "_", "L", "_", "L", "_", "O"]
    T = len(sequence)
    V = blank + 1
    post = np.full((T, V), 1e-6, dtype=np.float32)
    for t, ch in enumerate(sequence):
        if ch == "_":
            post[t, blank] = 1.0 - 1e-6 * (V - 1)
        else:
            post[t, charset.index(ch)] = 1.0 - 1e-6 * (V - 1)
    post /= post.sum(axis=1, keepdims=True)
    result = decoder.decode_one(post)
    # The middle "L _ L" must decode to "LL" (blank separates, then collapse)
    assert result == "HALLO"


def test_lexicon_decoder_beats_greedy_on_noisy_posteriors():
    """With noise that would make greedy decoding produce a non-lexicon
    string, the lexicon decoder must still recover a valid lexicon word."""
    rng = np.random.default_rng(0)
    lexicon = ["HALLO", "HILFE", "BERLIN", "PYTHON"]
    decoder = W.LexiconDecoder(lexicon, beam_width=8)
    charset = decoder.charset
    blank = len(charset)

    # Noisy posteriors that spell HALLO with blanks between chars (CTC convention)
    # Sequence: H _ A _ L _ L _ O with noise
    seq = ["H", "_", "A", "_", "L", "_", "L", "_", "O"]
    T = len(seq)
    V = blank + 1
    post = rng.dirichlet(np.ones(V), size=T).astype(np.float32) * 0.05
    for i, ch in enumerate(seq):
        if ch == "_":
            post[i, blank] = 0.95
        else:
            post[i, charset.index(ch)] = 0.95
    post /= post.sum(axis=1, keepdims=True)

    result = decoder.decode_one(post)
    # The lexicon decoder must produce a valid lexicon word
    assert result in lexicon, f"decoded {result!r} not in lexicon"


# --------------------------------------------------------------------------- #
# Synthetic dataset fixture (mirrors OnHW-Words500 layout)
# --------------------------------------------------------------------------- #
@pytest.fixture
def words500_dataset_dir():
    """A tiny synthetic words500 fold directory, named as the archives name it."""
    rng = np.random.default_rng(0)
    lexicon = ["HALLO", "WELT", "TEST", "BERLIN"]
    d = tempfile.mkdtemp(prefix="onhw_words500_")
    fold_dir = os.path.join(d, "0")
    os.makedirs(fold_dir, exist_ok=True)

    # Train: 8 samples (2 per word), Val: 4 samples (1 per word)
    n_train, n_val = 8, 4
    X_train = [
        rng.normal(0, 1, size=(50 + i, 13)).astype(np.float32) for i in range(n_train)
    ]
    X_val = [
        rng.normal(0, 1, size=(50 + i, 13)).astype(np.float32) for i in range(n_val)
    ]
    # Train: each word twice; Val: each word once
    Y_train = [W.encode_word(lexicon[i % 4]) for i in range(n_train)]
    Y_val = [W.encode_word(lexicon[i % 4]) for i in range(n_val)]
    train_ids = [i % 2 for i in range(n_train)]  # 2 writers
    val_ids = [i % 2 for i in range(n_val)]

    for fname, obj in [
        ("all_x_dat_train_imu.pkl", X_train),
        ("all_x_dat_val_imu.pkl", X_val),
        ("all_train_gt.pkl", Y_train),
        ("all_val_gt.pkl", Y_val),
        ("train_ids.pkl", train_ids),
        ("val_ids.pkl", val_ids),
    ]:
        with open(os.path.join(fold_dir, fname), "wb") as f:
            pickle.dump(obj, f)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_load_words500_basic(words500_dataset_dir):
    ds = W.load_onhw_words500(words500_dataset_dir, fold=0)
    assert ds.format == "words500_pkl"
    assert ds.fold == 0
    assert ds.n_train == 8
    assert ds.n_val == 4
    assert ds.n_writers == 2
    assert ds.vocab_size == 59
    # All sequences have 13 channels
    assert all(s.shape[1] == 13 for s in ds.X_train)
    # Words are decoded correctly
    assert "HALLO" in ds.train_words
    assert "WELT" in ds.val_words


def test_load_words500_lexicon(words500_dataset_dir):
    """The lexicon property returns the sorted unique words across train+val."""
    ds = W.load_onhw_words500(words500_dataset_dir, fold=0)
    lex = ds.lexicon
    assert lex == sorted(set(["HALLO", "WELT", "TEST", "BERLIN"]))


def test_load_words500_invalid_fold_raises(words500_dataset_dir):
    with pytest.raises(ValueError):
        W.load_onhw_words500(words500_dataset_dir, fold=5)


def test_load_words500_missing_fold_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="fold 0 not found"):
        W.load_onhw_words500(str(tmp_path), fold=0)


def test_load_words500_missing_pkl_raises(tmp_path):
    """If a required .pkl file is missing, the loader raises FileNotFoundError."""
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    # Create only one of the six required files
    (fold_dir / "all_x_dat_train_imu.pkl").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="missing"):
        W.load_onhw_words500(str(tmp_path), fold=0)


def test_load_words500_summary_string(words500_dataset_dir):
    ds = W.load_onhw_words500(words500_dataset_dir, fold=0)
    s = ds.summary()
    assert "fold 0" in s
    assert "train=8" in s
    assert "val=4" in s
    assert "lexicon=4 words" in s


# --------------------------------------------------------------------------- #
# Demo mode (no real data needed)
# --------------------------------------------------------------------------- #
def test_demo_runs_without_errors(capsys):
    """The --demo entry point must run end-to-end without raising."""
    W._demo()
    out = capsys.readouterr().out
    assert "Greedy decode:" in out
    assert "Lexicon decode:" in out


# --------------------------------------------------------------------------- #
# Archive layout - regression tests for the real OnHW-Words500 releases
#
# These pin the layout verified against OnHW-Words500_indep_L.zip (2022-05-20).
# The loader originally assumed `fold_0` directories and rejected the blank
# index as an out-of-range token, so it could not read the shipped archives
# at all; both assumptions are pinned here.
# --------------------------------------------------------------------------- #
def _write_fold(fold_dir, label_seqs, pad_to=None):
    """Write one fold in the shipped layout, right-padding labels like the archives."""
    os.makedirs(fold_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    n = len(label_seqs)
    if pad_to is not None:
        label_seqs = [
            list(s) + [W.WORDS500_BLANK_IDX] * (pad_to - len(s)) for s in label_seqs
        ]
    x = [rng.normal(size=(30, 13)).astype(np.float32) for _ in range(n)]
    payload = {
        "all_x_dat_train_imu.pkl": x,
        "all_x_dat_val_imu.pkl": x,
        "all_train_gt.pkl": label_seqs,
        "all_val_gt.pkl": label_seqs,
        "train_ids.pkl": [1052] * n,
        "val_ids.pkl": [4003] * n,
    }
    for fname, obj in payload.items():
        with open(os.path.join(fold_dir, fname), "wb") as f:
            pickle.dump(obj, f)


def test_loader_accepts_the_bare_integer_fold_directory(tmp_path):
    """The shipped archives name folds `0`..`4`, not `fold_0`."""
    _write_fold(str(tmp_path / "0"), [W.encode_word("Juni")], pad_to=19)
    ds = W.load_onhw_words500(str(tmp_path), fold=0)
    assert ds.train_words == ["Juni"]


def test_loader_still_accepts_a_renamed_fold_directory(tmp_path):
    """`fold_0` stays supported so manually renamed extracts keep working."""
    _write_fold(str(tmp_path / "fold_0"), [W.encode_word("Juni")], pad_to=19)
    assert W.load_onhw_words500(str(tmp_path), fold=0).train_words == ["Juni"]


def test_missing_fold_error_lists_what_is_actually_there(tmp_path):
    (tmp_path / "3").mkdir()
    with pytest.raises(FileNotFoundError, match=r"Subdirectories present: \['3'\]"):
        W.load_onhw_words500(str(tmp_path), fold=0)


def test_label_padding_is_stripped(tmp_path):
    """Archives right-pad labels to a fixed width with the blank index.

    A CTC target that carries that padding is longer than the real word and
    makes the loss meaningless, so the loader has to remove it.
    """
    words = ["gerade", "Juni", "immer", "Ich"]
    _write_fold(str(tmp_path / "0"), [W.encode_word(w) for w in words], pad_to=19)
    ds = W.load_onhw_words500(str(tmp_path), fold=0)
    assert [len(y) for y in ds.Y_train] == [len(w) for w in words]
    assert ds.train_words == words
    assert all(W.WORDS500_BLANK_IDX not in y for y in ds.Y_train)


def test_blank_index_is_a_legal_token(tmp_path):
    """The range check must allow the blank, or every real archive is rejected."""
    _write_fold(str(tmp_path / "0"), [W.encode_word("Ich")], pad_to=19)
    ds = W.load_onhw_words500(str(tmp_path), fold=0)  # must not raise
    assert ds.n_train == 1


def test_a_token_above_the_blank_is_still_rejected(tmp_path):
    _write_fold(str(tmp_path / "0"), [[0, 1, W.WORDS500_BLANK_IDX + 7]])
    with pytest.raises(ValueError, match="different charset"):
        W.load_onhw_words500(str(tmp_path), fold=0)


def test_umlauts_round_trip_through_the_charset():
    """The German words in this dataset need ÄÖÜäöüß; 26+26+7 = 59 symbols."""
    for word in ("Fräulein", "Öl", "Straße", "Übung", "Jahr"):
        assert W.decode_tokens(W.encode_word(word)) == word


# --------------------------------------------------------------------------- #
# LexiconDecoder
# --------------------------------------------------------------------------- #
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK = len(ALPHA)


def _posteriors(path, sharp=0.99):
    """One posterior row per entry of `path` (a list of token indices)."""
    p = np.full((len(path), len(ALPHA) + 1), (1.0 - sharp) / len(ALPHA))
    for t, tok in enumerate(path):
        p[t, tok] = sharp
    return p / p.sum(axis=1, keepdims=True)


def test_decoder_never_returns_a_non_lexicon_prefix():
    """Closed vocabulary: 'BE' is not an answer, even if the model stops there.

    The model emits B, blank, E and then only blanks. A prefix-pruning decoder
    that scores partial prefixes returns 'BE'; a closed-vocabulary decoder has
    to commit to a real word.
    """
    p = _posteriors([ALPHA.index("B"), BLANK, ALPHA.index("E"), BLANK, BLANK])
    out = W.LexiconDecoder(["BEAM", "BERLIN"], charset=ALPHA).decode_one(p)
    assert out in {"BEAM", "BERLIN"}
    assert out != "BE"


def test_decoder_handles_a_doubled_letter():
    """'ALL' is only reachable when the second L extends a blank-ending path.

    Collapsing each prefix to a single best alignment loses that path, so
    doubled letters silently become single ones.
    """
    p = _posteriors(
        [ALPHA.index("A"), BLANK, ALPHA.index("L"), BLANK, ALPHA.index("L"), BLANK]
    )
    assert W.LexiconDecoder(["ALL", "ALARM"], charset=ALPHA).decode_one(p) == "ALL"


def test_decoder_collapses_repeats_without_an_intervening_blank():
    """A, L, L with no blank between the Ls is CTC for 'AL', not 'ALL'."""
    p = _posteriors([ALPHA.index("A"), ALPHA.index("L"), ALPHA.index("L")])
    assert W.LexiconDecoder(["AL", "ALL"], charset=ALPHA).decode_one(p) == "AL"


def test_decoder_sums_over_alignments_rather_than_taking_the_best_one():
    """A word reachable many ways must beat one reachable a single sharp way.

    'AA' has many alignments across 6 frames of diffuse A/blank mass while
    'BB' needs one precise route; maximising over alignments can pick 'BB',
    summing cannot.
    """
    p = np.full((6, len(ALPHA) + 1), 1e-4)
    p[:, ALPHA.index("A")] = 0.30
    p[:, BLANK] = 0.30
    p[0, ALPHA.index("B")] = 0.34
    p[2, ALPHA.index("B")] = 0.34
    p = p / p.sum(axis=1, keepdims=True)
    assert (
        W.LexiconDecoder(["AA", "BB"], charset=ALPHA, beam_width=16).decode_one(p)
        == "AA"
    )


def test_decoder_rejects_posteriors_of_the_wrong_width():
    dec = W.LexiconDecoder(["AB"], charset=ALPHA)
    with pytest.raises(ValueError, match="expected 27"):
        dec.decode_one(np.full((4, 12), 1 / 12))


def test_decoder_picks_the_right_word_from_a_large_lexicon():
    lexicon = ["HALLO", "WELT", "PYTHON", "BEAM", "DECODE", "WORD", "TEST"]
    dec = W.LexiconDecoder(lexicon, charset=ALPHA, beam_width=8)
    for word in lexicon:
        path = []
        for ch in word:
            path += [ALPHA.index(ch), BLANK]
        assert dec.decode_one(_posteriors(path)) == word


def test_decoder_is_deterministic():
    p = _posteriors(
        [
            ALPHA.index("W"),
            BLANK,
            ALPHA.index("E"),
            BLANK,
            ALPHA.index("L"),
            BLANK,
            ALPHA.index("T"),
        ]
    )
    dec = W.LexiconDecoder(["WELT", "WORD"], charset=ALPHA)
    assert len({dec.decode_one(p) for _ in range(5)}) == 1


def test_empty_lexicon_yields_an_empty_decode():
    p = _posteriors([ALPHA.index("A"), BLANK])
    assert W.LexiconDecoder([], charset=ALPHA).decode_one(p) == ""


def test_strict_reports_a_miss_rather_than_a_non_word():
    """No lexicon word is reachable, so strict mode must not invent one."""
    p = _posteriors([ALPHA.index("Q"), BLANK, ALPHA.index("Z")])
    assert W.LexiconDecoder(["BEAM"], charset=ALPHA, strict=True).decode_one(p) == ""


def test_non_strict_falls_back_to_the_best_prefix():
    p = _posteriors([ALPHA.index("B"), BLANK, ALPHA.index("E"), BLANK, BLANK])
    out = W.LexiconDecoder(
        ["BEAM"], charset=ALPHA, strict=False, beam_width=2
    ).decode_one(p)
    assert out == "" or "BEAM".startswith(out)


def test_strict_still_returns_a_word_when_one_is_reachable():
    p = _posteriors(
        [
            ALPHA.index("B"),
            BLANK,
            ALPHA.index("E"),
            BLANK,
            ALPHA.index("A"),
            BLANK,
            ALPHA.index("M"),
        ]
    )
    assert (
        W.LexiconDecoder(["BEAM"], charset=ALPHA, strict=True).decode_one(p) == "BEAM"
    )
