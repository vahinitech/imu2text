"""Tests for the CTC sequence-to-sequence pipeline (onhw_seq2seq)."""

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
