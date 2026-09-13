"""Tests for the CTC sequence-to-sequence pipeline (onhw_seq2seq)."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorflow")

# pylint: disable=wrong-import-position
# The skip above has to run first: the module pulls in keras at import
# time, so importing it earlier would fail the whole file rather than
# skip it when TensorFlow is missing.
from imu2text import seq2seq as S  # noqa: E402


def test_edit_distance():
    assert S.edit_distance("kitten", "sitting") == 3
    assert S.edit_distance("", "abc") == 3
    assert S.edit_distance("abc", "abc") == 0
    assert S.edit_distance(["a", "b"], ["b"]) == 1
    assert S.edit_distance(["a", "b"], ["c", "d"]) == 2


def test_cer_wer():
    refs, hyps = ["12+3=15", "7-2=5"], ["12+3=15", "7-2=6"]
    assert S.cer(refs, hyps) == pytest.approx(1 / 12)
    assert S.wer(["the quick fox"], ["the quik fox"]) == pytest.approx(1 / 3)


def test_charset_roundtrip():
    cs = S.Charset(["12+3", "7-2", "0="])
    assert cs.size == len(set("12+37-20="))
    for lab in ["12+3", "7-2", "0="]:
        assert cs.decode(cs.encode(lab)) == lab


def test_ctc_model_trains_and_decodes():
    """One tiny training run on synthetic motifs: loss drops, decode works."""
    x, labels = S.make_demo_data(n=48, seed=0)
    charset = S.Charset(labels)
    train_idx = np.arange(len(x))
    maxlen = 160
    X, Y, label_len = S.prepare(x, labels, charset, maxlen, train_idx)
    train_model, infer_model, down_len = S.build_ctc_models(
        maxlen, charset.size, rnn_units=16, rnn_layers=1
    )
    assert down_len == maxlen // 4
    assert down_len >= label_len.max()

    input_len = np.full((len(x), 1), down_len, dtype=np.int32)
    dummy = np.zeros((len(x), 1), dtype=np.float32)
    hist = train_model.fit(
        [X, Y, input_len, label_len], dummy, epochs=2, batch_size=16, verbose=0
    )
    losses = hist.history["loss"]
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0]  # CTC loss decreases even in 2 epochs

    hyps = S.ctc_greedy_decode(infer_model, X[:4], down_len, charset)
    assert len(hyps) == 4
    assert all(isinstance(h, str) for h in hyps)
    assert all(all(ch in charset.symbols for ch in h) for h in hyps)


def test_padding_does_not_change_recurrent_outputs():
    """Frames beyond the receptive field of valid input cannot affect the BiLSTM."""
    _, model, _ = S.build_ctc_models(80, 2, rnn_units=4, rnn_layers=1)
    X = np.ones((2, 80, 13), dtype=np.float32)
    X[1, 40:] = 100
    output = model.predict([X, np.array([[4], [4]])], verbose=0)
    np.testing.assert_allclose(output[0, :4], output[1, :4], atol=1e-6)


def test_batch_trimming_preserves_last_valid_convolution_context():
    """Trimming must retain the partial pool and both convolutions' right context."""
    _, model, _ = S.build_ctc_models(80, 2, rnn_units=4, rnn_layers=1)
    X = np.zeros((1, 80, 13), dtype=np.float32)
    X[0, :31] = np.random.default_rng(5).normal(size=(31, 13))
    X[0, 30] = 20
    lengths = np.array([[7]], dtype=np.int32)
    arrays = [X, np.array([[0]]), lengths, np.array([[1]])]
    inputs, _ = S.CTCBatches(arrays, [0], 1)[0]
    full = model.predict([X, lengths], verbose=0)
    trimmed = model.predict([inputs[0], lengths], verbose=0)
    np.testing.assert_allclose(full[:, :7], trimmed[:, :7], atol=1e-6, rtol=1e-6)


def test_greedy_decode_ignores_predictions_after_actual_length():
    class Predictor:
        def predict(self, inputs, verbose=0):
            del verbose
            assert np.array_equal(inputs[1], [[1], [2]])
            return np.array([[[0.99, 0.005, 0.005], [0.005, 0.99, 0.005]]] * 2)

    hyps = S.ctc_greedy_decode(
        Predictor(), np.zeros((2, 8, 13)), [1, 2], S.Charset(["ab"])
    )
    assert hyps == ["a", "ab"]


def test_minibatches_preserve_every_sample_and_are_repeatable():
    arrays = [
        np.arange(7).reshape(-1, 1, 1),
        np.arange(7).reshape(-1, 1) + 100,
        np.ones((7, 1), dtype=np.int32),
        np.ones((7, 1), dtype=np.int32),
    ]
    first = S.CTCBatches(arrays, np.arange(7), 3, seed=8)
    second = S.CTCBatches(arrays, np.arange(7), 3, seed=8)
    for _ in range(2):
        seen = []
        for index in range(len(first)):
            inputs, dummy = first[index]
            np.testing.assert_array_equal(inputs[0], second[index][0][0])
            np.testing.assert_array_equal(inputs[1], inputs[0][:, :, 0] + 100)
            assert dummy.shape == inputs[1].shape
            seen.extend(inputs[0].ravel())
        assert sorted(seen) == list(range(7))
        first.on_epoch_end()
        second.on_epoch_end()


def test_sequence_scaler_does_not_fit_held_out_recordings():
    train = np.arange(104, dtype=np.float32).reshape(8, 13)
    small, _, _ = S.prepare([train, train], ["a", "a"], S.Charset(["a"]), 8, [0])
    large, _, _ = S.prepare([train, train * 1e6], ["a", "a"], S.Charset(["a"]), 8, [0])
    np.testing.assert_array_equal(small[0], large[0])


@pytest.mark.parametrize("test_writer", [1, 3])
def test_words_cli_preserves_the_archives_writer_protocol(monkeypatch, test_writer):
    from imu2text import words

    sample = np.ones((80, 13), dtype=np.float32)
    ds = SimpleNamespace(
        X_train=[sample, sample],
        X_val=[sample],
        train_words=["a", "a"],
        val_words=["a"],
        train_ids=np.array([1, 2]),
        val_ids=np.array([test_writer]),
        n_train=2,
        n_val=1,
        n_writers=len({1, 2, test_writer}),
        lexicon=["a"],
    )
    captured = {}
    monkeypatch.setattr(words, "load_onhw_words500", lambda *a, **k: ds)
    monkeypatch.setattr(S, "run", lambda *a, **k: captured.update(k))
    monkeypatch.setattr("sys.argv", ["seq2seq", "--onhw-words500", "unused"])
    S.main()
    assert captured["n_train"] == 2
    if test_writer == 1:
        assert captured["writers"] is None
    else:
        np.testing.assert_array_equal(captured["writers"], [1, 2, 3])
