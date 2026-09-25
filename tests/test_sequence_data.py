"""Regression checks for sequence alignment and evaluation partitions."""

import numpy as np
import pytest

from imu2text.sequence_data import (
    decoder_lengths,
    resample_bounds,
    sequence_split,
    validate_ctc_lengths,
)


def test_official_test_is_preserved_and_validation_writers_are_disjoint():
    writers = np.repeat(np.arange(10), 4)
    train, val, test = sequence_split(40, seed=0, n_train=32, writers=writers)
    assert np.array_equal(test, np.arange(32, 40))
    assert set(writers[train]).isdisjoint(writers[val])
    assert set(writers[train]).isdisjoint(writers[test])
    assert sorted(np.concatenate([train, val, test])) == list(range(40))


def test_writer_dependent_archive_cannot_be_reported_as_writer_independent():
    with pytest.raises(ValueError, match="disjoint test writers"):
        sequence_split(12, 0, n_train=8, writers=np.tile([0, 1], 6))


def test_resampling_keeps_word_endings_and_uses_fixed_bounds():
    short = np.arange(12, dtype=np.float32).reshape(6, 2)
    long = np.arange(400, dtype=np.float32).reshape(200, 2)
    unchanged = np.ones((40, 2), dtype=np.float32)
    samples = resample_bounds([short, long, unchanged], 20, 80)
    assert [len(sample) for sample in samples] == [20, 80, 40]
    for source, result in zip([short, long, unchanged], samples):
        np.testing.assert_array_equal(result[0], source[0])
        np.testing.assert_array_equal(result[-1], source[-1])


def test_ctc_requires_blank_frames_for_adjacent_repeats():
    validate_ctc_lengths(["ab", "aa"], [2, 3])
    with pytest.raises(ValueError, match="adjacent repeats"):
        validate_ctc_lengths(["aa"], [2])


@pytest.mark.parametrize("values", [[1, 0], [1], [1, 2.5]])
def test_decoder_rejects_invalid_lengths(values):
    with pytest.raises(ValueError, match="positive integer"):
        decoder_lengths(values, 2)


def test_decoder_accepts_scalar_and_per_recording_lengths():
    np.testing.assert_array_equal(decoder_lengths(3, 2), [3, 3])
    np.testing.assert_array_equal(decoder_lengths([[1], [3]], 2), [1, 3])
