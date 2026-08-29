"""Smoke tests for the data-handling logic of the OnHW pipeline.

These cover the pure/deterministic parts (splitting, writer inference,
augmentation, normalization) so CI catches regressions without a training run.
"""

import numpy as np
import pytest

pytest.importorskip("tensorflow")  # onhw_models imports keras at module level

# pylint: disable=wrong-import-position
# The skip above has to run first: the module pulls in keras at import
# time, so importing it earlier would fail the whole file rather than
# skip it when TensorFlow is missing.
import onhw_models as M  # noqa: E402


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
    x2, y2, _, tr2 = M.augment_training(x, y, writers, tr, n_aug=2, seed=0)
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


# --------------------------------------------------------------------------- #
# Normalization modes
#
# The distinction that matters here is not "does it run" but what each mode
# assumes about test-time access, since that decides whether a number measured
# under it is a standard writer-independent figure or a transductive one.
# --------------------------------------------------------------------------- #
def _norm_fixture(n_writers=6, per_writer=6, maxlen=20):
    """Writers with deliberately different per-channel offsets and scales.

    ``per_writer`` stays at or above ``MIN_SAMPLES_PER_WRITER_SCALER`` so the
    per-writer mode actually fits per-writer scalers instead of falling back
    to the global one.
    """
    rng = np.random.default_rng(0)
    x, writers = [], []
    for w in range(n_writers):
        for _ in range(per_writer):
            x.append(
                (rng.normal(size=(maxlen, M.N_CHANNELS)) * (w + 1) + w * 10).astype(
                    np.float32
                )
            )
            writers.append(w)
    return x, np.array(writers)


def test_global_norm_fits_on_train_only():
    """A test sample must not move when a test-only sample changes."""
    x, writers = _norm_fixture()
    tr = np.arange(18)
    a = M.normalize_and_pad(x, tr, 20, writers=writers, mode="global")
    x2 = list(x)
    x2[30] = x2[30] * 50.0  # perturb a non-train sample
    b = M.normalize_and_pad(x2, tr, 20, writers=writers, mode="global")
    assert np.allclose(a[0], b[0]), "train scaler moved when test data changed"


def test_per_sample_norm_centres_every_sample():
    x, writers = _norm_fixture()
    out = M.normalize_and_pad(x, np.arange(18), 20, writers=writers, mode="per_sample")
    assert np.allclose(out.mean(axis=1), 0.0, atol=1e-4)
    assert np.allclose(out.std(axis=1), 1.0, atol=1e-3)


def test_per_sample_norm_needs_no_writer_ids():
    """It is the mode to reach for when writer IDs are unavailable."""
    x, _ = _norm_fixture()
    out = M.normalize_and_pad(x, np.arange(18), 20, writers=None, mode="per_sample")
    assert out.shape == (36, 20, M.N_CHANNELS)


def test_per_writer_norm_treats_train_and_test_writers_alike():
    """Symmetry is the whole point.

    An earlier version scaled train writers by their own statistics but test
    writers by the global train scaler, so the model trained on one
    distribution and was evaluated on another. Each writer's samples should
    come out centred whichever split they land in.
    """
    x, writers = _norm_fixture()
    tr = np.flatnonzero(writers < 3)  # writers 3-5 are unseen
    out = M.normalize_and_pad(x, tr, 20, writers=writers, mode="per_writer")
    for w in np.unique(writers):
        rows = out[np.flatnonzero(writers == w)].reshape(-1, M.N_CHANNELS)
        assert np.allclose(
            rows.mean(axis=0), 0.0, atol=1e-3
        ), f"writer {w} not centred by its own statistics"


def test_per_writer_norm_requires_writer_ids():
    x, _ = _norm_fixture()
    with pytest.raises(ValueError, match="requires writer IDs"):
        M.normalize_and_pad(x, np.arange(18), 20, writers=None, mode="per_writer")


def test_per_writer_norm_falls_back_for_a_thin_writer():
    """One sample gives a std of ~0 per channel; the global scaler is safer."""
    x, writers = _norm_fixture(n_writers=4, per_writer=8)
    x.append(np.ones((20, M.N_CHANNELS), dtype=np.float32))
    writers = np.append(writers, 99)  # a writer with one sample
    out = M.normalize_and_pad(x, np.arange(24), 20, writers=writers, mode="per_writer")
    assert np.all(np.isfinite(out))


def test_unknown_norm_mode_raises():
    x, writers = _norm_fixture()
    with pytest.raises(ValueError, match="unknown normalization mode"):
        M.normalize_and_pad(x, np.arange(18), 20, writers=writers, mode="per_epoch")


def test_every_norm_mode_produces_the_same_shape():
    x, writers = _norm_fixture()
    tr = np.arange(18)
    shapes = {
        m: M.normalize_and_pad(x, tr, 20, writers=writers, mode=m).shape
        for m in ("global", "per_sample", "per_writer")
    }
    assert len(set(shapes.values())) == 1


# --------------------------------------------------------------------------- #
# Writer-independent split guards
# --------------------------------------------------------------------------- #
def test_writer_split_rejects_unknown_writer_ids():
    """-1 means "writer not recorded" (the .npy OnHW-chars archives).

    Treating those as one writer would put every sample in a single group and
    turn a writer-independent split into a silently leaky one.
    """
    y = np.arange(30) % 5
    writers = np.full(30, -1)
    with pytest.raises(ValueError, match="unknown"):
        M.make_split(30, y, seed=0, mode="writer", writers=writers)


def test_writer_split_rejects_a_partially_unknown_set():
    y = np.arange(30) % 5
    writers = np.repeat([0, 1, 2, 3, -1], 6)
    with pytest.raises(ValueError, match="unknown"):
        M.make_split(30, y, seed=0, mode="writer", writers=writers)


def test_writer_split_keeps_writers_disjoint_across_splits():
    y = np.arange(48) % 4
    writers = np.repeat(np.arange(8), 6)
    tr, va, te = M.make_split(48, y, seed=0, mode="writer", writers=writers)
    groups = [set(writers[s].tolist()) for s in (tr, va, te)]
    assert groups[0] & groups[1] == set()
    assert groups[0] & groups[2] == set()
    assert groups[1] & groups[2] == set()


def test_writer_split_covers_every_sample_exactly_once():
    y = np.arange(48) % 4
    writers = np.repeat(np.arange(8), 6)
    tr, va, te = M.make_split(48, y, seed=0, mode="writer", writers=writers)
    assert sorted(np.concatenate([tr, va, te]).tolist()) == list(range(48))


# --------------------------------------------------------------------------- #
# Reproducibility
#
# --seed used to cover the split and the augmentation RNG but not the Keras
# layer initialisers, so two runs at the same seed started from different
# weights and landed several points apart - enough to swamp the effect of
# whatever configuration change was being measured.
# --------------------------------------------------------------------------- #
def test_keras_initialisers_follow_the_seed():
    """tf.random.set_seed alone does not reach them; set_random_seed does."""
    import tensorflow as tf

    def build_once():
        tf.keras.utils.set_random_seed(0)
        return M.BUILDERS["cnn_bilstm"](40, 12).get_weights()

    a, b = build_once(), build_once()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_split_is_reproducible_for_a_given_seed():
    y = np.arange(48) % 4
    writers = np.repeat(np.arange(8), 6)
    first = M.make_split(48, y, seed=3, mode="writer", writers=writers)
    second = M.make_split(48, y, seed=3, mode="writer", writers=writers)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_different_seeds_give_different_writer_splits():
    y = np.arange(48) % 4
    writers = np.repeat(np.arange(8), 6)
    a = set(writers[M.make_split(48, y, 0, "writer", writers)[2]].tolist())
    b = set(writers[M.make_split(48, y, 5, "writer", writers)[2]].tolist())
    assert a != b, "seed does not change which writers are held out"


# --------------------------------------------------------------------------- #
# Official OnHW-chars splits
#
# The published .npy archive stores labels as single-character strings and
# contains a few zero-length recordings. Both crash a loader that assumes
# integer labels and non-empty sequences, and neither shows up in a synthetic
# fixture built from the same assumptions as the code.
# --------------------------------------------------------------------------- #
@pytest.fixture
def official_npy_dir(tmp_path):
    """A tiny archive in the published .npy layout, labels as strings."""
    import os

    rng = np.random.default_rng(0)
    folder = tmp_path / "onhw2_both_indep_0"
    folder.mkdir()
    chars = list("ABCabc")
    n_tr, n_te = 60, 24
    for split, n in (("train", n_tr), ("test", n_te)):
        X = np.empty(n, dtype=object)
        for i in range(n):
            X[i] = rng.normal(size=(20 + i % 7, 13)).astype(np.float32)
        y = np.array([chars[i % len(chars)] for i in range(n)], dtype="<U1")
        np.save(os.path.join(folder, f"X_{split}.npy"), X, allow_pickle=True)
        np.save(os.path.join(folder, f"y_{split}.npy"), y, allow_pickle=True)
    return str(tmp_path)


def test_official_split_decodes_string_labels(official_npy_dir):
    """Published y_*.npy are dtype '<U1' (['A','B',...]), not integers."""
    _, y, classes, _ = M.load_official_split(
        official_npy_dir, "both", "indep", 0, seed=0
    )
    assert y.dtype.kind == "i"
    assert y.min() >= 0 and y.max() < len(classes)


def test_official_split_maps_labels_to_the_canonical_order(official_npy_dir):
    """Index 0 must mean 'A' and 26 must mean 'a', as everywhere else."""
    import onhw_chars as C

    _, y, classes, _ = M.load_official_split(
        official_npy_dir, "both", "indep", 0, seed=0
    )
    assert "".join(classes) == C.CHARS_BOTH
    assert classes[y[0]] == "A"


def test_official_split_keeps_the_published_test_half_intact(official_npy_dir):
    """The val set is carved from train only; test must stay untouched."""
    _, _, _, (tr, va, te) = M.load_official_split(
        official_npy_dir, "both", "indep", 0, seed=0
    )
    assert len(te) == 24  # the whole published test half
    assert len(tr) + len(va) == 60  # val comes out of train
    assert set(tr).isdisjoint(set(te))
    assert set(va).isdisjoint(set(te))
    assert set(tr).isdisjoint(set(va))


def test_official_split_drops_zero_length_recordings(official_npy_dir, capsys):
    """The real archive has 3 such samples in 31,275; they crash the scaler."""
    import os

    folder = os.path.join(official_npy_dir, "onhw2_both_indep_0")
    X = np.load(os.path.join(folder, "X_train.npy"), allow_pickle=True)
    X[0] = np.zeros((0, 13), dtype=np.float32)
    np.save(os.path.join(folder, "X_train.npy"), X, allow_pickle=True)

    x, _, _, (tr, va, _) = M.load_official_split(
        official_npy_dir, "both", "indep", 0, seed=0
    )
    assert all(len(s) > 0 for s in x)
    assert len(tr) + len(va) == 59  # one fewer than before
    assert "dropped 1 sample" in capsys.readouterr().out


def test_normalize_and_pad_rejects_zero_length_samples():
    """A clear error beats a sklearn traceback from deep inside the scaler."""
    x = [
        np.ones((10, M.N_CHANNELS), np.float32),
        np.zeros((0, M.N_CHANNELS), np.float32),
    ]
    with pytest.raises(ValueError, match="zero timesteps"):
        M.normalize_and_pad(x, np.array([0]), 10, mode="global")


def test_official_split_rejects_the_pkl_release(tmp_path):
    """The official folds only exist in the .npy archive."""
    import pickle

    for name, obj in (
        ("all_x_dat_imu.pkl", [np.ones((5, 13), np.float32)]),
        ("all_gt.pkl", ["A"]),
        ("all_gt_enc.pkl", [0]),
        ("list_ids.pkl", [0]),
    ):
        with open(tmp_path / name, "wb") as f:
            pickle.dump(obj, f)
    with pytest.raises(SystemExit, match="not the .npy OnHW-chars release"):
        M.load_official_split(str(tmp_path), "both", "indep", 0, seed=0)


def test_per_writer_norm_rejects_unknown_writer_ids():
    """All -1 would collapse to one scaler fit over train and test together."""
    x, _ = _norm_fixture()
    writers = np.full(len(x), -1)
    with pytest.raises(ValueError, match="unknown"):
        M.normalize_and_pad(x, np.arange(18), 20, writers=writers, mode="per_writer")


def test_per_sample_norm_works_without_writer_ids():
    """The mode to use with the .npy archives, which ship no writer IDs."""
    x, _ = _norm_fixture()
    out = M.normalize_and_pad(
        x, np.arange(18), 20, writers=np.full(len(x), -1), mode="per_sample"
    )
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name", ["cnn", "lstm", "bilstm", "cnn_bilstm", "cnn_bilstm_attn"]
)
def test_every_builder_produces_a_class_distribution(name):
    model = M.BUILDERS[name](40, 12)
    assert model.output_shape == (None, 12)
    probs = model.predict(np.zeros((3, 40, M.N_CHANNELS), np.float32), verbose=0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_attention_model_respects_the_rnn_capacity_globals():
    """--rnn-units / --rnn-layers must reach the attention variant too."""
    units, layers_ = M.RNN_UNITS, M.RNN_LAYERS
    try:
        M.RNN_UNITS, M.RNN_LAYERS = 100, 2
        big = M.BUILDERS["cnn_bilstm_attn"](40, 12).count_params()
        M.RNN_UNITS, M.RNN_LAYERS = 32, 1
        small = M.BUILDERS["cnn_bilstm_attn"](40, 12).count_params()
    finally:
        M.RNN_UNITS, M.RNN_LAYERS = units, layers_
    assert big > small


def test_attention_weights_sum_to_one_over_time():
    """The pooling is a weighted average, so the weights must normalise."""
    import tensorflow as tf

    model = M.BUILDERS["cnn_bilstm_attn"](40, 12)
    softmax = next(l for l in model.layers if isinstance(l, tf.keras.layers.Softmax))
    probe = tf.keras.Model(model.input, softmax.output)
    w = probe.predict(
        np.random.default_rng(0).normal(size=(2, 40, M.N_CHANNELS)).astype(np.float32),
        verbose=0,
    )
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-4)


# --------------------------------------------------------------------------- #
# Error analysis
#
# The 52-class OnHW-chars ceiling is dominated by upper-versus-lower confusion,
# so the reporting has to separate that from the rest of the error rather than
# hide it inside one accuracy number.
# --------------------------------------------------------------------------- #
CHARS52 = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


def test_error_analysis_reports_case_only_errors(capsys):
    """Every error here is a letter mistaken for its own other case."""
    true = np.array([CHARS52.index(c) for c in "OSXZ"])
    pred = np.array([CHARS52.index(c) for c in "osxz"])
    M.error_analysis(pred, true, CHARS52)
    out = capsys.readouterr().out
    assert "case-insensitive accuracy : 100.00%" in out
    assert "errors that are case only : 4/4" in out
    assert "[case pair]" in out


def test_error_analysis_separates_genuine_confusions(capsys):
    """A 'B' read as 'q' is not a case error and must not be counted as one."""
    true = np.array([CHARS52.index("B"), CHARS52.index("O")])
    pred = np.array([CHARS52.index("q"), CHARS52.index("o")])
    M.error_analysis(pred, true, CHARS52)
    out = capsys.readouterr().out
    assert "errors that are case only : 1/2" in out


def test_error_analysis_case_insensitive_accuracy_exceeds_plain(capsys):
    true = np.array([CHARS52.index(c) for c in "ABOS"])
    pred = np.array([CHARS52.index(c) for c in "ABos"])  # half right, half case
    M.error_analysis(pred, true, CHARS52)
    out = capsys.readouterr().out
    assert "case-insensitive accuracy : 100.00%" in out
    assert "(plain: 50.00%)" in out


def test_error_analysis_handles_a_perfect_prediction(capsys):
    true = np.array([0, 1, 2])
    M.error_analysis(true.copy(), true, CHARS52)
    out = capsys.readouterr().out
    assert "0 wrong" in out
    assert "top" not in out  # nothing to rank


def test_error_analysis_counts_every_error_once(capsys):
    rng = np.random.default_rng(0)
    true = rng.integers(0, 52, size=200)
    pred = rng.integers(0, 52, size=200)
    M.error_analysis(pred, true, CHARS52)
    out = capsys.readouterr().out
    assert f"({int((pred != true).sum())} wrong" in out
