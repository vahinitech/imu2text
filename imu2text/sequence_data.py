"""Length handling and writer partitions for sequence recognition."""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def sequence_split(n, seed, n_train=None, writers=None):
    """Keep the official test partition and hold out training writers for validation."""
    idx = np.arange(n)
    if n_train is None:
        if writers is not None:
            raise ValueError("writer metadata requires an explicit training partition")
        train, tmp = train_test_split(idx, test_size=0.4, random_state=seed)
        val, test = train_test_split(tmp, test_size=0.5, random_state=seed)
        return train, val, test
    if not 1 < n_train < n:
        raise ValueError("n_train must leave nonempty training and test partitions")
    test = idx[n_train:]
    if writers is None:
        train, val = train_test_split(idx[:n_train], test_size=0.15, random_state=seed)
    else:
        writers = np.asarray(writers)
        if writers.shape != (n,) or np.any(writers < 0):
            raise ValueError("a known writer ID is required for every recording")
        if not set(writers[:n_train]).isdisjoint(writers[n_train:]):
            raise ValueError(
                "writer-independent evaluation requires disjoint test writers"
            )
        groups = writers[:n_train]
        if len(np.unique(groups)) < 2:
            raise ValueError("at least two training writers are needed for validation")
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        train, val = next(splitter.split(idx[:n_train], groups=groups))
    return train, val, test


def resample_bounds(x, minlen, maxlen):
    """Resample outside fixed duration bounds without consulting target labels.

    Long recordings retain their whole signal instead of losing the word ending.
    Short recordings get more alignment frames, but no new sensor information.
    """
    if not 4 <= minlen <= maxlen:
        raise ValueError("require 4 <= min_len <= max_len")
    out = []
    for sample in x:
        sample = np.asarray(sample, dtype=np.float32)
        if sample.ndim != 2 or not len(sample) or not np.isfinite(sample).all():
            raise ValueError(
                "recordings must be nonempty finite (time, channels) arrays"
            )
        size = int(np.clip(len(sample), minlen, maxlen))
        if size != len(sample):
            source = np.linspace(0, 1, len(sample))
            target = np.linspace(0, 1, size)
            sample = np.stack(
                [np.interp(target, source, channel) for channel in sample.T], axis=1
            ).astype(np.float32)
        out.append(sample)
    return out


def validate_ctc_lengths(labels, lengths):
    """CTC needs an extra blank frame between adjacent equal target symbols."""
    required = np.array(
        [len(label) + sum(a == b for a, b in zip(label, label[1:])) for label in labels]
    )
    lengths = np.asarray(lengths).reshape(-1)
    if lengths.shape != required.shape or np.any(lengths < 1):
        raise ValueError("one positive output length is required per label")
    invalid = lengths < required
    if np.any(invalid):
        raise ValueError(
            f"{int(invalid.sum())} targets cannot align: CTC needs label length plus "
            "adjacent repeats; adjust fixed duration bounds or pooling before training"
        )


def decoder_lengths(lengths, n):
    """Accept one output length per recording, or a scalar for legacy callers."""
    values = np.asarray(lengths)
    if values.ndim == 0:
        values = np.full(n, values.item())
    values = values.reshape(-1)
    if (
        values.shape != (n,)
        or np.any(values < 1)
        or np.any(values != values.astype(int))
    ):
        raise ValueError("one positive integer output length is required per recording")
    return values.astype(np.int32)
