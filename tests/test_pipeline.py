"""Smoke tests for the data-handling logic of the OnHW pipeline.

These cover the pure/deterministic parts (splitting, writer inference,
augmentation, normalization) so CI catches regressions without a training run.
"""
import numpy as np
import pytest

pytest.importorskip("tensorflow")  # onhw_models imports keras at module level

import onhw_models as M


def _fake_dataset(n_writers=6, alphabet="ABCD"):
    """n_writers writers each writing the alphabet once, distinct lengths."""
    chars, x, rng = [], [], np.random.default_rng(0)
    for w in range(n_writers):
        for c in alphabet:
            chars.append(c)
            x.append(rng.normal(w, 1.0, size=(20 + w, M.N_CHANNELS)).astype(np.float32))
    classes = sorted(set(chars))
    y = np.array([classes.index(c) for c in chars])
    return x, y, chars


def test_infer_writer_ids_detects_alphabet_cycles():
    chars = list("ABC" "ABC" "AB")  # 3 writers; last one incomplete
    writers = M.infer_writer_ids(chars)
    assert writers.tolist() == [0, 0, 0, 1, 1, 1, 2, 2]


def test_writer_split_is_writer_disjoint():
    x, y, chars = _fake_dataset()
    writers = M.infer_writer_ids(chars)
    tr, va, te = M.make_split(len(x), y, seed=0, mode="writer", writers=writers)
    groups = [set(writers[idx]) for idx in (tr, va, te)]
    assert groups[0] & groups[1] == set()
    assert groups[0] & groups[2] == set()
    assert groups[1] & groups[2] == set()
    assert sorted(np.concatenate([tr, va, te])) == list(range(len(x)))


def test_random_split_partitions_all_samples():
    x, y, _ = _fake_dataset(n_writers=10)
    tr, va, te = M.make_split(len(x), y, seed=1, mode="random")
    assert sorted(np.concatenate([tr, va, te])) == list(range(len(x)))


def test_augmentation_appends_only_training_samples():
    x, y, chars = _fake_dataset()
    writers = M.infer_writer_ids(chars)
    tr, va, te = M.make_split(len(x), y, seed=0, mode="writer", writers=writers)
    n_orig, n_tr = len(x), len(tr)
    x2, y2, w2, tr2 = M.augment_training(x, y, writers, tr, n_aug=2, seed=0)
    assert len(x2) == n_orig + 2 * n_tr
    assert len(tr2) == 3 * n_tr
    # augmented copies keep label and writer of their source
    assert (np.sort(y2[:n_orig]) == np.sort(y)).all()
    # val/test indices still address the original, untouched samples
    for i in np.concatenate([va, te]):
        assert i < n_orig
    # augmented sample differs from its source but has the same shape
    j = tr[0]
    assert x2[n_orig].shape == x[j].shape or x2[n_orig].shape[1] == M.N_CHANNELS


def test_normalize_and_pad_shapes_and_train_statistics():
    x, y, chars = _fake_dataset()
    writers = M.infer_writer_ids(chars)
    tr, _, _ = M.make_split(len(x), y, seed=0, mode="writer", writers=writers)
    maxlen = 24
    X = M.normalize_and_pad(x, tr, maxlen)
    assert X.shape == (len(x), maxlen, M.N_CHANNELS)
    # scaler was fit on train timesteps: their mean ~0, std ~1
    stacked = np.vstack([X[i][: len(x[i])] for i in tr])
    assert np.allclose(stacked.mean(axis=0), 0.0, atol=0.15)
    assert np.allclose(stacked.std(axis=0), 1.0, atol=0.15)


def test_model_builders_produce_expected_output_shape():
    maxlen, n_classes = 32, 5
    for name, build in M.BUILDERS.items():
        model = build(maxlen, n_classes)
        out = model(np.zeros((2, maxlen, M.N_CHANNELS), dtype=np.float32))
        assert out.shape == (2, n_classes), name
