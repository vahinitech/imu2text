"""Integration tests against the real Fraunhofer OnHW archives.

Every test here skips unless the archives are already extracted, so CI stays
offline and fast. Point ``ONHW_DATA_DIR`` at a directory holding the extracted
folders to run them:

    python onhw_download.py onhw_chars_L onhw_symbols_L onhw_words500_indep_L --out ./data
    ONHW_DATA_DIR=./data python -m pytest tests/test_real_data.py -v

These exist because the synthetic fixtures elsewhere encode the same
assumptions as the loaders, so they cannot catch a loader that disagrees with
the published archives. Two such disagreements were real: the Words500 fold
directories are named ``0``..``4`` rather than ``fold_0``, and its labels are
right-padded with the blank index. Both are asserted below against the actual
files.
"""

import os

import numpy as np
import pytest

DATA_DIR = os.environ.get("ONHW_DATA_DIR")

pytestmark = pytest.mark.skipif(
    not DATA_DIR or not os.path.isdir(DATA_DIR),
    reason="set ONHW_DATA_DIR to a directory of extracted OnHW archives",
)


def _require(*candidates):
    """Return the first extracted archive folder that exists, else skip."""
    for name in candidates:
        path = os.path.join(DATA_DIR, name)
        if os.path.isdir(path):
            return path
    return pytest.skip(f"none of {candidates} found under {DATA_DIR}")


# --------------------------------------------------------------------------- #
# OnHW-chars_L (3.5 MB)
# --------------------------------------------------------------------------- #
def test_real_chars_L_loads_with_the_documented_shape():
    import onhw_chars as C

    ds = C.load_onhw_chars(_require("OnHW-chars_L"))
    assert ds.format == "pkl"
    assert ds.n_samples == 2270  # as published
    assert ds.n_classes == 52  # A-Z + a-z
    assert ds.n_writers == 9
    assert ds.has_writer_ids


def test_real_chars_L_channels_are_13_wide():
    import onhw_chars as C

    ds = C.load_onhw_chars(_require("OnHW-chars_L"))
    assert all(s.shape[1] == C.N_CHANNELS for s in ds.X_all[:200])


def test_real_chars_L_writer_ids_are_remapped_contiguously():
    """The archive stores recording IDs (…, 1052, 4003), not 0..8."""
    import onhw_chars as C

    ds = C.load_onhw_chars(_require("OnHW-chars_L"))
    assert sorted(set(ds.writers.tolist())) == list(range(9))


def test_real_chars_L_labels_follow_the_canonical_class_order():
    import onhw_chars as C

    ds = C.load_onhw_chars(_require("OnHW-chars_L"))
    assert "".join(ds.classes) == C.CHARS_BOTH
    assert ds.y_all.min() == 0 and ds.y_all.max() == 51


def test_real_chars_L_supports_a_writer_independent_split():
    """The whole point of keeping real writer IDs."""
    pytest.importorskip("tensorflow")
    import onhw_chars as C
    import onhw_models as M

    ds = C.load_onhw_chars(_require("OnHW-chars_L"))
    tr, va, te = M.make_split(
        ds.n_samples, ds.y_all, seed=0, mode="writer", writers=ds.writers
    )
    train_w = set(ds.writers[tr].tolist())
    assert train_w.isdisjoint(set(ds.writers[te].tolist()))
    assert len(tr) and len(va) and len(te)


# --------------------------------------------------------------------------- #
# OnHW-symbols / equations (7.5 MB)
# --------------------------------------------------------------------------- #
def test_real_symbols_L_has_15_classes():
    import onhw_symbols as S

    ds = S.load_onhw_symbols(_require("OnHW-symbols_equations_L"))
    assert ds.n_classes == len(S.SYMBOLS_VOCAB) == 15
    labels = np.asarray(ds.y_train)
    assert labels.min() >= 0 and labels.max() <= 14


def test_real_symbols_L_ships_no_split():
    """The left-handed archive has no train/val split; that must be visible."""
    import onhw_symbols as S

    ds = S.load_onhw_symbols(_require("OnHW-symbols_equations_L"))
    assert ds.has_official_split is False
    assert ds.n_val == 0
    assert "no split shipped" in ds.summary()


def test_real_symbols_dep_uses_the_shipped_split():
    """The dep archive is flat with an official 1853/473 split and no folds."""
    import onhw_symbols as S

    ds = S.load_onhw_symbols(_require("OnHW-symbols_equations_dep"))
    assert ds.has_official_split
    assert ds.n_train == 1853 and ds.n_val == 473
    assert ds.n_train + ds.n_val == 2326  # as published
    assert ds.n_writers == 27


def test_real_symbols_dep_is_writer_dependent_as_its_name_says():
    """All 27 writers appear on both sides; reporting it as WI would be wrong."""
    import onhw_symbols as S

    ds = S.load_onhw_symbols(_require("OnHW-symbols_equations_dep"))
    assert ds.is_writer_independent is False
    assert set(ds.train_ids.tolist()) == set(ds.val_ids.tolist())


def test_real_symbols_dep_labels_span_the_charset():
    import onhw_symbols as S

    ds = S.load_onhw_symbols(_require("OnHW-symbols_equations_dep"))
    assert set(np.unique(ds.y_train).tolist()) == set(range(15))


# --------------------------------------------------------------------------- #
# OnHW-Words500 (14 MB for the left-handed writer-independent archive)
# --------------------------------------------------------------------------- #
def test_real_words500_loads_from_a_bare_integer_fold_directory():
    """The published archives name folds `0`/`1`, not `fold_0`."""
    import onhw_words as W

    ds = W.load_onhw_words500(_require("Words500_indep_L", "Words500_dep_L"), fold=0)
    assert ds.n_train > 0 and ds.n_val > 0


def test_real_words500_labels_have_their_padding_stripped():
    """Labels ship right-padded to a fixed width with the blank index."""
    import onhw_words as W

    ds = W.load_onhw_words500(_require("Words500_indep_L", "Words500_dep_L"), fold=0)
    assert all(W.WORDS500_BLANK_IDX not in y for y in ds.Y_train)
    assert len(set(len(y) for y in ds.Y_train)) > 1, "labels still fixed-width"


def test_real_words500_decodes_to_german_words():
    import onhw_words as W

    ds = W.load_onhw_words500(_require("Words500_indep_L", "Words500_dep_L"), fold=0)
    assert all(w and w.strip() for w in ds.train_words)
    assert {"gerade", "Juni", "immer"} <= set(ds.train_words)


def test_real_words500_lexicon_is_the_closed_500_word_vocabulary():
    import onhw_words as W

    ds = W.load_onhw_words500(_require("Words500_indep_L", "Words500_dep_L"), fold=0)
    assert len(ds.lexicon) == 500


def test_real_words500_is_writer_independent_across_splits():
    """The `indep` archive must not share a writer between train and val."""
    import onhw_words as W

    base = _require("Words500_indep_L")
    ds = W.load_onhw_words500(base, fold=0)
    assert set(ds.train_ids.tolist()).isdisjoint(set(ds.val_ids.tolist()))


def test_real_words500_folds_swap_the_writers():
    """Fold 1 is fold 0 with train and val exchanged (2 writers, 2 folds)."""
    import onhw_words as W

    base = _require("Words500_indep_L")
    f0 = W.load_onhw_words500(base, fold=0)
    f1 = W.load_onhw_words500(base, fold=1)
    assert set(f0.train_ids.tolist()) == set(f1.val_ids.tolist())


def test_real_words500_lexicon_decoder_recovers_a_known_word():
    """End-to-end: the real 500-word lexicon drives the decoder."""
    import onhw_words as W

    ds = W.load_onhw_words500(_require("Words500_indep_L", "Words500_dep_L"), fold=0)
    decoder = W.LexiconDecoder(ds.lexicon, beam_width=8)

    target = ds.train_words[0]
    blank = W.WORDS500_BLANK_IDX
    path = []
    for ch in target:
        path += [W.WORDS500_VOCAB.index(ch), blank]
    post = np.full((len(path), blank + 1), 1e-4)
    for t, tok in enumerate(path):
        post[t, tok] = 0.99
    post /= post.sum(axis=1, keepdims=True)

    assert decoder.decode_one(post) == target
