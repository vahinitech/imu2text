"""Tests for the IMU-specific augmentation transforms in ``onhw_augment``.

These cover the deterministic invariants (output shape, magnitude
preservation for rotations, no-leak for ``augment_training``) so CI catches
regressions without needing TensorFlow.
"""

import numpy as np
import pytest

from imu2text import augment as A


@pytest.fixture
def seq():
    """A (60, 13) fake IMU sample with distinct magnitudes per triad."""
    rng = np.random.default_rng(0)
    s = np.empty((60, 13), dtype=np.float32)
    # accel in g (~1), gyro in deg/s (~100), mag in uT (~50), force in N (~1)
    s[:, 0:3] = rng.normal(0, 1, size=(60, 3))
    s[:, 3:6] = rng.normal(0, 1, size=(60, 3))
    s[:, 6:9] = rng.normal(0, 100, size=(60, 3))
    s[:, 9:12] = rng.normal(0, 50, size=(60, 3))
    s[:, 12] = rng.normal(1, 0.5, size=60)
    return s


def test_jitter_preserves_shape(seq):
    out = A.jitter(seq, np.random.default_rng(1))
    assert out.shape == seq.shape


def test_per_channel_scale_preserves_shape(seq):
    out = A.per_channel_scale(seq, np.random.default_rng(2))
    assert out.shape == seq.shape


def test_mag_warp_preserves_shape(seq):
    out = A.mag_warp(seq, np.random.default_rng(3))
    assert out.shape == seq.shape


def test_time_warp_preserves_shape(seq):
    out = A.time_warp(seq, np.random.default_rng(4))
    assert out.shape == seq.shape


def test_random_rotation_preserves_vector_magnitudes(seq):
    """A rotation must preserve the magnitude of each 3-vector at every step."""
    R = A._random_rotation_matrix(np.random.default_rng(5), max_angle_deg=30.0)
    # Check R is a valid rotation: orthonormal, det = +1
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)

    # Applying it to each triad preserves per-row vector norms
    out = A.random_rotation(seq, np.random.default_rng(6), max_angle_deg=15.0)
    for cols in A.TRIAD_GROUPS:
        before = np.linalg.norm(seq[:, cols], axis=1)
        after = np.linalg.norm(out[:, cols], axis=1)
        assert np.allclose(before, after, atol=1e-4), f"triad {cols} magnitudes changed"


def test_random_rotation_leaves_force_untouched(seq):
    """The Force channel is a scalar - rotations must not touch it."""
    out = A.random_rotation(seq, np.random.default_rng(7), max_angle_deg=15.0)
    assert np.allclose(out[:, 12], seq[:, 12])


def test_channel_dropout_keeps_force(seq):
    """Force channel signals pen-on-paper contact - never drop it."""
    rng = np.random.default_rng(8)
    for _ in range(20):  # try many draws so dropout has chances to fire
        out = A.channel_dropout(seq, rng, p_drop=0.5)
        assert np.allclose(out[:, 12], seq[:, 12]), "force channel was dropped"
        # at most half the channels dropped
        n_dropped = int(np.sum(~np.all(out == seq, axis=0)))
        assert n_dropped <= seq.shape[1] // 2


def test_random_crop_preserves_length(seq):
    """Crop output must match input length (padding back when shorter)."""
    out = A.random_crop(seq, np.random.default_rng(9), min_frac=0.5)
    assert out.shape == seq.shape


def test_augment_one_shape_dtype(seq):
    out = A.augment_one(seq, np.random.default_rng(10))
    assert out.shape == seq.shape
    assert out.dtype == np.float32


def test_augment_training_appends_only_training_samples():
    """Same contract as the legacy augment_training: val/test indices untouched."""
    n = 30
    rng = np.random.default_rng(0)
    x = [rng.normal(0, 1, size=(20, 13)).astype(np.float32) for _ in range(n)]
    y = np.arange(n) % 5
    writers = np.arange(n) // 5
    tr = np.arange(20)
    va = np.arange(20, 25)
    te = np.arange(25, 30)
    x2, _, _, tr2 = A.augment_training(x, y, writers, tr, n_aug=3, seed=0)
    assert len(x2) == n + 3 * len(tr)
    assert len(tr2) == 4 * len(tr)
    # val/test indices still address original, untouched samples
    for i in np.concatenate([va, te]):
        assert i < n
    # augmented sample differs from its source (rotation/warp/noise applied)
    assert not np.allclose(x2[n], x[tr[0]])


def test_augmentation_config_defaults_match_legacy():
    """Defaults must reproduce the legacy 64.8 -> 71.6 jump documented in README."""
    cfg = A.AugmentationConfig()
    assert cfg.jitter_sigma == 0.05
    assert cfg.scale_sigma == 0.08
    assert cfg.mag_warp_sigma == 0.15
    assert cfg.time_warp_sigma == 0.2
    # new defaults are sensible (not 0, not extreme)
    assert 5.0 <= cfg.rotation_max_deg <= 15.0
    assert 0.0 < cfg.channel_dropout_p < 0.1
    assert 0.8 <= cfg.crop_min_frac < 1.0


def test_augment_one_zero_config_is_identity(seq):
    """A policy with all probabilities 0 must return the input unchanged."""
    cfg = A.AugmentationConfig(
        p_jitter=0.0,
        p_scale=0.0,
        p_mag_warp=0.0,
        p_time_warp=0.0,
        p_rotation=0.0,
        p_channel_dropout=0.0,
        p_crop=0.0,
        rotation_max_deg=0.0,
        channel_dropout_p=0.0,
    )
    out = A.augment_one(seq, np.random.default_rng(11), cfg)
    assert np.allclose(out, seq, atol=1e-6)


# --------------------------------------------------------------------------- #
# Policies
#
# The default policy has to stay bit-for-bit the one behind the measured 71.6%
# writer-independent result. Enabling the newer transforms by default would
# change what `--augment N` does and quietly invalidate that number, so these
# tests pin which transforms each policy turns on.
# --------------------------------------------------------------------------- #
def test_legacy_is_the_default_policy():
    assert A.AugmentationConfig() == A.AugmentationConfig.legacy()


def test_legacy_leaves_the_unmeasured_transforms_off():
    cfg = A.AugmentationConfig.legacy()
    assert cfg.p_rotation == 0.0
    assert cfg.p_channel_dropout == 0.0
    assert cfg.p_crop == 0.0


def test_legacy_keeps_the_measured_transforms_on():
    cfg = A.AugmentationConfig.legacy()
    assert cfg.p_jitter == 1.0 and cfg.p_scale == 1.0
    assert cfg.p_mag_warp == 0.7 and cfg.p_time_warp == 0.7


def test_extended_turns_the_new_transforms_on():
    cfg = A.AugmentationConfig.extended()
    assert cfg.p_rotation > 0 and cfg.p_channel_dropout > 0 and cfg.p_crop > 0


def test_extended_keeps_the_legacy_sigmas():
    legacy, extended = A.AugmentationConfig.legacy(), A.AugmentationConfig.extended()
    for field in ("jitter_sigma", "scale_sigma", "mag_warp_sigma", "time_warp_sigma"):
        assert getattr(legacy, field) == getattr(extended, field)


def test_aug_policies_registry_matches_the_classmethods():
    assert A.AUG_POLICIES["legacy"]() == A.AugmentationConfig.legacy()
    assert A.AUG_POLICIES["extended"]() == A.AugmentationConfig.extended()


def test_legacy_never_rotates(seq):
    """A rotation would break the per-triad magnitudes; legacy must not apply one."""
    cfg = A.AugmentationConfig.legacy()
    rng = np.random.default_rng(0)
    # Ratios between triad channels stay fixed under scale/jitter-free settings,
    # so isolate rotation by disabling everything else.
    cfg = A.AugmentationConfig(
        p_jitter=0.0,
        p_scale=0.0,
        p_mag_warp=0.0,
        p_time_warp=0.0,
        p_rotation=cfg.p_rotation,
        p_channel_dropout=0.0,
        p_crop=0.0,
    )
    for _ in range(30):
        assert np.allclose(A.augment_one(seq, rng, cfg), seq, atol=1e-5)


def test_extended_does_rotate(seq):
    """Guards against the extended policy silently degrading to legacy."""
    cfg = A.AugmentationConfig(
        p_jitter=0.0,
        p_scale=0.0,
        p_mag_warp=0.0,
        p_time_warp=0.0,
        p_rotation=1.0,
        p_channel_dropout=0.0,
        p_crop=0.0,
    )
    out = A.augment_one(seq, np.random.default_rng(1), cfg)
    assert not np.allclose(out, seq, atol=1e-5)
    for cols in A.TRIAD_GROUPS:  # but magnitudes survive
        assert np.allclose(
            np.linalg.norm(seq[:, cols], axis=1),
            np.linalg.norm(out[:, cols], axis=1),
            atol=1e-3,
        )


def test_augment_training_honours_the_policy():
    rng = np.random.default_rng(0)
    x = [rng.normal(size=(20, 13)).astype(np.float32) for _ in range(6)]
    y, writers, tr = np.arange(6) % 3, np.arange(6) // 3, np.arange(4)
    for cfg in (A.AugmentationConfig.legacy(), A.AugmentationConfig.extended()):
        x2, _, _, tr2 = A.augment_training(x, y, writers, tr, 2, seed=0, cfg=cfg)
        assert len(x2) == len(x) + 2 * len(tr)
        assert len(tr2) == 3 * len(tr)


# --------------------------------------------------------------------------- #
# Channel-count robustness
#
# Other pens have more than 13 channels, and downstream slices can be narrower.
# The triad-aware transforms index fixed columns, so they need explicit bounds.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_channels", [3, 6, 9, 12, 13, 16])
def test_transforms_survive_other_channel_counts(n_channels):
    seq = np.random.default_rng(0).normal(size=(24, n_channels)).astype(np.float32)
    rng = np.random.default_rng(1)
    for fn in (
        A.jitter,
        A.per_channel_scale,
        A.mag_warp,
        A.time_warp,
        A.random_rotation,
        A.channel_dropout,
        A.random_crop,
    ):
        assert fn(seq, rng).shape == seq.shape, f"{fn.__name__} changed shape"


def test_channel_dropout_does_not_index_past_a_narrow_input():
    """Force sits at column 12; a 6-channel sample has no such column."""
    seq = np.ones((10, 6), dtype=np.float32)
    assert (
        A.channel_dropout(seq, np.random.default_rng(0), p_drop=0.5).shape == seq.shape
    )


def test_rotation_leaves_extra_channels_alone():
    """A 16-channel sample keeps columns 12..15 untouched."""
    seq = np.random.default_rng(0).normal(size=(20, 16)).astype(np.float32)
    out = A.random_rotation(seq, np.random.default_rng(1), max_angle_deg=20.0)
    assert np.allclose(out[:, 12:], seq[:, 12:])


def test_augment_one_is_reproducible_from_a_seed(seq):
    cfg = A.AugmentationConfig.extended()
    a = A.augment_one(seq, np.random.default_rng(7), cfg)
    b = A.augment_one(seq, np.random.default_rng(7), cfg)
    assert np.array_equal(a, b)


def test_random_crop_keeps_length_across_fractions(seq):
    rng = np.random.default_rng(0)
    for frac in (0.5, 0.7, 0.85, 0.99):
        assert A.random_crop(seq, rng, min_frac=frac).shape == seq.shape
