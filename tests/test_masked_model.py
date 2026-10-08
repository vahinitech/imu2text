"""The masked CNN+BiLSTM+attention model ignores zero padding."""

import numpy as np

from imu2text import models as M


def _padded(seq, maxlen):
    out = np.zeros((1, maxlen, seq.shape[1]), dtype=np.float32)
    out[0, : len(seq)] = seq
    return out


def test_masked_model_output_does_not_depend_on_padding_length():
    seq = np.random.default_rng(0).normal(size=(40, M.N_CHANNELS)).astype(np.float32)
    short = M.build_cnn_bilstm_attn_masked(48, 5)
    long = M.build_cnn_bilstm_attn_masked(100, 5)
    long.set_weights(short.get_weights())
    a = short(_padded(seq, 48), training=False).numpy()
    b = long(_padded(seq, 100), training=False).numpy()
    assert np.allclose(a, b, atol=1e-5)


def test_unmasked_model_does_depend_on_padding_length():
    """The reason for the masked variant: padding changes the plain model."""
    seq = np.random.default_rng(0).normal(size=(40, M.N_CHANNELS)).astype(np.float32)
    short = M.build_cnn_bilstm_attn(48, 5)
    long = M.build_cnn_bilstm_attn(100, 5)
    long.set_weights(short.get_weights())
    a = short(_padded(seq, 48), training=False).numpy()
    b = long(_padded(seq, 100), training=False).numpy()
    assert not np.allclose(a, b, atol=1e-5)
