"""Tests for the OnHW-symbols and OnHW-equations loaders + transfer learning."""

import os
import pickle

import numpy as np
import pytest

from imu2text import symbols as S


# --------------------------------------------------------------------------- #
# Symbol vocabulary tests
# --------------------------------------------------------------------------- #
def test_symbols_vocab_size_is_15():
    """15 = 10 digits + 5 operators (+ - · : =)."""
    assert len(S.SYMBOLS_VOCAB) == 15
    assert S.SYMBOLS_VOCAB[:10] == "0123456789"
    assert S.SYMBOLS_VOCAB[10:] == "+-·:="
    assert S.SYMBOLS_BLANK_IDX == 15


def test_encode_decode_symbol_roundtrip():
    for s in "0123456789+-·:=":
        idx = S.encode_symbol(s)
        assert S.decode_symbol(idx) == s


def test_encode_symbol_raises_on_unknown():
    with pytest.raises(ValueError):
        S.encode_symbol("X")
    with pytest.raises(ValueError):
        S.encode_symbol("ab")


def test_decode_symbol_seq():
    """Decode a sequence of indices to a string."""
    # "12+3=15"
    seq = [S.encode_symbol(c) for c in "12+3=15"]
    assert S.decode_symbol_seq(seq) == "12+3=15"
    # Invalid indices (negatives, out-of-range) are dropped
    assert S.decode_symbol_seq([-1, 0, 99, 1]) == "01"


# --------------------------------------------------------------------------- #
# Synthetic dataset fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def symbols_dir_official(tmp_path):
    """An archive in the dep/indep layout: flat, with a shipped train/val split.

    Mirrors OnHW-symbols_equations_dep (verified 2021-09-02): no fold
    subfolders, six files per sub-dataset, writers shared across the split.
    """
    rng = np.random.default_rng(0)
    d = tmp_path / "OnHW-symbols_equations_dep"
    d.mkdir()
    for suffix in ("s", "e"):
        for split, n in (("train", 12), ("val", 6)):
            x = [rng.normal(size=(20, 13)).astype(np.float32) for _ in range(n)]
            y = [i % 15 for i in range(n)]
            ids = [1001 + (i % 3) for i in range(n)]  # same writers both sides
            for name, obj in (
                (f"all_x_dat_{split}_imu_{suffix}.pkl", x),
                (f"all_{split}_gt_{suffix}.pkl", y),
                (f"{split}_ids_{suffix}.pkl", ids),
            ):
                with open(d / name, "wb") as f:
                    pickle.dump(obj, f)
    return str(d)


@pytest.fixture
def symbols_dir_unsplit(tmp_path):
    """An archive in the left-handed layout: flat, and with no split at all."""
    rng = np.random.default_rng(1)
    d = tmp_path / "OnHW-symbols_equations_L"
    d.mkdir()
    for suffix in ("s", "e"):
        x = [rng.normal(size=(20, 13)).astype(np.float32) for _ in range(9)]
        y = [i % 15 for i in range(9)]
        ids = [1004 + (i % 2) for i in range(9)]
        for name, obj in (
            (f"all_x_dat_imu_{suffix}.pkl", x),
            (f"all_gt_{suffix}.pkl", y),
            (f"list_ids_{suffix}.pkl", ids),
        ):
            with open(d / name, "wb") as f:
                pickle.dump(obj, f)
    return str(d)


# --------------------------------------------------------------------------- #
# The shipped split is used as-is
#
# The dep archive deliberately shares writers between train and val and the
# indep archive deliberately does not. Re-deriving a split here would change
# which protocol a reported number belongs to, so these pin that the loader
# returns what the archive shipped.
# --------------------------------------------------------------------------- #
def test_official_split_is_returned_verbatim(symbols_dir_official):
    ds = S.load_onhw_symbols(symbols_dir_official)
    assert ds.has_official_split
    assert ds.n_train == 12 and ds.n_val == 6


def test_writer_dependent_archive_is_reported_as_writer_dependent(symbols_dir_official):
    """Writers are shared, so this must not be labelled writer-independent."""
    ds = S.load_onhw_symbols(symbols_dir_official)
    assert ds.is_writer_independent is False
    assert "writer-dependent" in ds.summary()


def test_writer_independent_archive_is_detected(tmp_path):
    rng = np.random.default_rng(2)
    d = tmp_path / "OnHW-symbols_equations_indep"
    d.mkdir()
    for split, n, writers in (("train", 12, [1001, 1002]), ("val", 6, [1003])):
        x = [rng.normal(size=(20, 13)).astype(np.float32) for _ in range(n)]
        for name, obj in (
            (f"all_x_dat_{split}_imu_s.pkl", x),
            (f"all_{split}_gt_s.pkl", [i % 15 for i in range(n)]),
            (f"{split}_ids_s.pkl", [writers[i % len(writers)] for i in range(n)]),
        ):
            with open(d / name, "wb") as f:
                pickle.dump(obj, f)
    ds = S.load_onhw_symbols(str(d))
    assert ds.is_writer_independent is True
    assert "writer-independent" in ds.summary()


def test_unsplit_archive_is_flagged_rather_than_silently_empty(symbols_dir_unsplit):
    """An empty val set is a trap unless the caller is told it is empty."""
    ds = S.load_onhw_symbols(symbols_dir_unsplit)
    assert ds.has_official_split is False
    assert ds.n_train == 9 and ds.n_val == 0
    assert "no split shipped" in ds.summary()


def test_unsplit_archive_is_not_called_writer_independent(symbols_dir_unsplit):
    """With no val split there is nothing to be independent of."""
    assert S.load_onhw_symbols(symbols_dir_unsplit).is_writer_independent is False


def test_loader_reads_the_official_layout_file_names(symbols_dir_official):
    """Regression: the loader once looked for all_x_dat_imu_s.pkl only.

    The dep/indep archives name that file all_x_dat_train_imu_s.pkl, so the
    loader could not open them at all.
    """
    assert os.path.exists(
        os.path.join(symbols_dir_official, "all_x_dat_train_imu_s.pkl")
    )
    assert not os.path.exists(os.path.join(symbols_dir_official, "all_x_dat_imu_s.pkl"))
    assert S.load_onhw_symbols(symbols_dir_official).n_train == 12


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="directory not found"):
        S.load_onhw_symbols(str(tmp_path / "nope"))


def test_directory_without_any_recognised_pickles_raises(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="no OnHW-symbols pickles"):
        S.load_onhw_symbols(str(d))


def test_labels_stay_inside_the_15_class_charset(symbols_dir_official):
    ds = S.load_onhw_symbols(symbols_dir_official)
    assert ds.y_train.min() >= 0 and ds.y_train.max() < len(S.SYMBOLS_VOCAB)


def test_equations_use_the_official_split_too(symbols_dir_official):
    ds = S.load_onhw_equations(symbols_dir_official)
    assert ds.n_train == 12 and ds.n_val == 6
    assert ds.split == "official"


def test_equations_unsplit_layout(symbols_dir_unsplit):
    ds = S.load_onhw_equations(symbols_dir_unsplit)
    assert ds.n_train == 9 and ds.n_val == 0


def test_equations_decode_to_charset_symbols(symbols_dir_official):
    ds = S.load_onhw_equations(symbols_dir_official)
    assert all(w in S.SYMBOLS_VOCAB for w in ds.train_words)


# --------------------------------------------------------------------------- #
# Transfer learning helper tests (require TF, skipped without)
# --------------------------------------------------------------------------- #
@pytest.fixture
def tiny_pretrained_model():
    """Build a tiny cnn_bilstm model to stand in for a pretrained chars model."""
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inp = layers.Input(shape=(20, 13))
    x = layers.Conv1D(8, 3, padding="same", activation="relu")(inp)
    x = layers.Bidirectional(layers.LSTM(8))(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(52, activation="softmax")(x)  # chars head
    model = Model(inp, out, name="fake_chars_model")
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model


def test_build_transfer_model_replaces_head(tiny_pretrained_model):
    """The transfer model has a 15-class head, not 52."""
    new_model = S.build_transfer_model(tiny_pretrained_model, n_classes=15)
    # The new head has 15 outputs
    dummy_input = np.zeros((2, 20, 13), dtype=np.float32)
    out = new_model.predict(dummy_input, verbose=0)
    assert out.shape == (2, 15)


def test_build_transfer_model_freezes_trunk(tiny_pretrained_model):
    """All layers except the new head are frozen initially."""
    new_model = S.build_transfer_model(tiny_pretrained_model, n_classes=15)
    # The last layer (new head) should be trainable
    assert new_model.layers[-1].trainable
    # All earlier layers should be frozen
    for layer in new_model.layers[:-1]:
        assert not layer.trainable, f"layer {layer.name} should be frozen"


def test_unfreeze_trunk_makes_all_layers_trainable(tiny_pretrained_model):
    """After unfreeze_trunk, every layer is trainable again."""
    new_model = S.build_transfer_model(tiny_pretrained_model, n_classes=15)
    S.unfreeze_trunk(new_model, lr=1e-4)
    for layer in new_model.layers:
        assert layer.trainable, f"layer {layer.name} should be trainable after unfreeze"


# --------------------------------------------------------------------------- #
# The transfer model must not disturb the model it was built from
#
# A functional Model built directly on another model's tensors shares its
# layer objects, so freezing or fine-tuning the transfer model would reach
# back into the pretrained chars model and change it in place.
# --------------------------------------------------------------------------- #
def test_transfer_model_shares_no_layers_with_the_pretrained_model(
    tiny_pretrained_model,
):
    new = S.build_transfer_model(tiny_pretrained_model, n_classes=15)
    shared = [
        a.name for a in new.layers for b in tiny_pretrained_model.layers if a is b
    ]
    assert shared == [], f"layers shared with the pretrained model: {shared}"


def test_transfer_model_leaves_the_pretrained_model_trainable(tiny_pretrained_model):
    before = [l.trainable for l in tiny_pretrained_model.layers]
    S.build_transfer_model(tiny_pretrained_model, n_classes=15)
    after = [l.trainable for l in tiny_pretrained_model.layers]
    assert before == after, "building the transfer model froze the original"


def test_fine_tuning_does_not_change_the_pretrained_weights(tiny_pretrained_model):
    import tensorflow as tf

    before = [w.copy() for w in tiny_pretrained_model.get_weights()]
    new = S.build_transfer_model(tiny_pretrained_model, n_classes=15)
    S.unfreeze_trunk(new)
    x = np.random.default_rng(0).random((24, 20, 13))
    y = tf.keras.utils.to_categorical(np.arange(24) % 15, 15)
    new.fit(x, y, epochs=2, verbose=0)
    assert all(
        np.array_equal(a, b)
        for a, b in zip(before, tiny_pretrained_model.get_weights())
    )


def test_transfer_model_copies_the_trunk_weights(tiny_pretrained_model):
    """Cloning must carry the learned weights over, not reinitialise them."""
    new = S.build_transfer_model(tiny_pretrained_model, n_classes=15)
    old_trunk = tiny_pretrained_model.layers[1].get_weights()
    new_trunk = new.layers[1].get_weights()
    assert all(np.allclose(a, b) for a, b in zip(old_trunk, new_trunk))


def test_transfer_model_rejects_a_model_that_is_too_shallow():
    pytest.importorskip("tensorflow")
    from tensorflow.keras import layers, Model

    inp = layers.Input(shape=(13,))
    tiny = Model(inp, layers.Dense(52, activation="softmax")(inp))  # 2 layers
    with pytest.raises(ValueError, match="expected a model ending"):
        S.build_transfer_model(tiny, n_classes=15)


def test_equations_dataset_exposes_the_same_split_properties(symbols_dir_official):
    """The equations tuple needs these too; models.py reads them on both."""
    ds = S.load_onhw_equations(symbols_dir_official)
    assert ds.has_official_split is True
    assert ds.is_writer_independent is False  # fixture shares writers


def test_equations_unsplit_archive_reports_no_split(symbols_dir_unsplit):
    ds = S.load_onhw_equations(symbols_dir_unsplit)
    assert ds.has_official_split is False
    assert ds.is_writer_independent is False
