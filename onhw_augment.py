"""IMU-specific data augmentation transforms for OnHW-style sensor sequences.

This module collects the augmentation transforms used by ``onhw_models.py``.
Splitting them out of the training script keeps the augmentation policy
visible and unit-testable, and lets the same transforms be reused by the
sequence-to-sequence pipeline (``onhw_seq2seq.py``) and any downstream
task that operates on the same 13-channel OnHW channel layout.

The OnHW pen produces 13 channels at every timestep:

    [0:3]   Acc1 X, Y, Z      front accelerometer (3-vector)
    [3:6]   Acc2 X, Y, Z      rear accelerometer  (3-vector)
    [6:9]   Gyro  X, Y, Z     gyroscope           (3-vector)
    [9:12]  Mag   X, Y, Z     magnetometer        (3-vector)
    [12]    Force             scalar

The first four blocks are *3-vectors* in their sensor's own frame. A small
random rotation of that frame is a physically meaningful augmentation: it
simulates the pen being held in a slightly different grip / orientation,
which is exactly the kind of variation a writer-independent model has to
cope with. Rotations preserve the magnitude of the vectors (so acceleration
norms, gyro angular rates, etc. are unchanged), they only redistribute the
energy across the three axes - which is the right kind of perturbation.

Transforms
----------
Each transform takes a ``(T, C)`` float array and returns a ``(T, C)``
float array of the same shape. Transforms are stochastic; pass a
``numpy.random.Generator`` to make them reproducible.

- ``jitter``         - Gaussian noise proportional to per-channel std.
- ``per_channel_scale`` - multiply each channel by a slightly different gain.
- ``mag_warp``       - smooth random multiplicative envelope (sensor drift).
- ``time_warp``      - smooth non-linear time reparameterization.
- ``random_rotation``- small 3D rotation applied to each Acc/Gyro/Mag triad.
- ``channel_dropout``- zero out a few channels for the whole sample.
- ``random_crop``    - random sub-window (padded back to the original length).

The high-level entry point is ``augment_one(seq, rng, cfg)`` which applies a
random subset of these transforms to one sequence. ``AugmentationConfig``
exposes every knob.

Two policies ship. ``legacy`` (the default) is jitter + per-channel scale +
magnitude warp + time warp - the exact policy behind the measured 64.8% ->
71.6% writer-independent jump on the bundled OnHW subset. ``extended`` adds
rotation, channel dropout and random crop; it is unmeasured here, so it is
opt-in rather than default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# Channel layout of the OnHW DigiPen (and the Vahini pen, which extends it
# with 3 extra channels at the end). All transforms operate on the first 13
# channels; extra channels at the end (e.g. Vahini's 16-channel variant) are
# passed through untouched by the triad-aware transforms.
SENSOR_GROUPS: Dict[str, List[int]] = {
    "acc1": [0, 1, 2],
    "acc2": [3, 4, 5],
    "gyro": [6, 7, 8],
    "mag": [9, 10, 11],
}
FORCE_IDX = 12
TRIAD_GROUPS = list(SENSOR_GROUPS.values())


# --------------------------------------------------------------------------- #
# Individual transforms
# --------------------------------------------------------------------------- #
def jitter(
    seq: np.ndarray, rng: np.random.Generator, sigma: float = 0.05
) -> np.ndarray:
    """Add Gaussian noise proportional to each channel's std.

    Acc / Gyro / Mag / Force channels have wildly different magnitudes
    (accelerometers in g, gyro in deg/s, force in N), so a fixed absolute
    sigma would over-augment some channels and under-augment others. We
    scale sigma by each channel's own std to keep the perturbation
    comparable across sensor types.
    """
    std = seq.std(axis=0, keepdims=True) + 1e-6
    noise = rng.normal(0.0, sigma, size=seq.shape).astype(seq.dtype, copy=False)
    return seq + noise * std.astype(seq.dtype, copy=False)


def per_channel_scale(
    seq: np.ndarray, rng: np.random.Generator, sigma: float = 0.08
) -> np.ndarray:
    """Multiply each channel by an independent Gaussian gain (sensor gain drift)."""
    gains = rng.normal(1.0, sigma, size=(1, seq.shape[1])).astype(seq.dtype, copy=False)
    return seq * gains


def mag_warp(
    seq: np.ndarray, rng: np.random.Generator, sigma: float = 0.15, knots: int = 4
) -> np.ndarray:
    """Multiply the signal by a smooth random envelope (sensor gain drift).

    The envelope is a piecewise-linear curve through ``knots+2`` random
    points; knot positions are fixed in normalized time, only the y-values
    are random. This is the standard "magnitude warping" of Um et al. (2017).
    """
    t = seq.shape[0]
    knot_x = np.linspace(0.0, 1.0, knots + 2)
    curve = np.interp(
        np.linspace(0.0, 1.0, t), knot_x, rng.normal(1.0, sigma, size=knots + 2)
    ).astype(seq.dtype, copy=False)
    return seq * curve[:, None]


def time_warp(
    seq: np.ndarray, rng: np.random.Generator, sigma: float = 0.2, knots: int = 4
) -> np.ndarray:
    """Locally speed up / slow down the trajectory (smooth, monotonic warp).

    A piecewise-linear random time map with fixed knot x-positions and
    Gaussian y-positions, then made strictly monotonic by cumulative-max.
    Each channel is then resampled by linear interpolation against this map.
    """
    t = seq.shape[0]
    base = np.linspace(0.0, 1.0, t)
    knot_x = np.linspace(0.0, 1.0, knots + 2)
    knot_y = knot_x + rng.normal(0.0, sigma / knots, size=knots + 2)
    knot_y[0], knot_y[-1] = 0.0, 1.0
    knot_y = np.maximum.accumulate(knot_y)  # keep time monotonic
    warped = np.interp(base, knot_x, knot_y)
    out = np.empty_like(seq)
    for c in range(seq.shape[1]):
        out[:, c] = np.interp(warped, base, seq[:, c])
    return out


def _random_rotation_matrix(
    rng: np.random.Generator, max_angle_deg: float
) -> np.ndarray:
    """A small random 3x3 rotation matrix.

    Built as R = Rz(c) * Ry(b) * Rx(a) with each Euler angle drawn uniformly
    from [-max_angle, +max_angle]. The result is orthonormal with det=+1, so
    applying it to a 3-vector preserves its magnitude - which is what we want
    for an IMU frame rotation (the accelerometer norm, gyro rate, etc. should
    not change, only how the energy splits across the three axes).
    """
    a, b, c = rng.uniform(-max_angle_deg, max_angle_deg, size=3) * np.pi / 180.0
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cc, sc = np.cos(c), np.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def random_rotation(
    seq: np.ndarray, rng: np.random.Generator, max_angle_deg: float = 10.0
) -> np.ndarray:
    """Apply an independent small 3D rotation to each Acc/Gyro/Mag triad.

    A different rotation is drawn for each triad because the four sensors sit
    at different positions/orientations inside the pen - a real grip change
    affects each sensor's frame slightly differently.

    The Force channel (index 12) and any extra channels past 13 are left
    untouched (rotations don't apply to a scalar).
    """
    out = seq.astype(seq.dtype, copy=True)
    for cols in TRIAD_GROUPS:
        if max(cols) >= seq.shape[1]:
            break  # fewer channels than 13, no more triads
        R = _random_rotation_matrix(rng, max_angle_deg).astype(seq.dtype)
        out[:, cols] = out[:, cols] @ R.T  # (T, 3) @ (3, 3) -> rotate each row
    return out


def channel_dropout(
    seq: np.ndarray, rng: np.random.Generator, p_drop: float = 0.05
) -> np.ndarray:
    """Zero out a few channels for the whole sample (sensor dropout).

    Simulates a sensor channel going dead for a whole recording - a rare but
    real failure mode of pen-mounted sensors. We drop channels independently
    with probability ``p_drop``, capping at half the channels so the sample
    stays informative. Always keeps the Force channel (it's a single scalar
    that signals pen-on-paper contact).
    """
    n_channels = seq.shape[1]
    drop = rng.random(n_channels) < p_drop
    if FORCE_IDX < n_channels:  # always keep the force channel
        drop[FORCE_IDX] = False
    if drop.sum() > n_channels // 2:  # never drop more than half
        drop[:] = False
    if drop.any():
        seq = seq.copy()
        seq[:, drop] = 0.0
    return seq


def random_crop(
    seq: np.ndarray, rng: np.random.Generator, min_frac: float = 0.85
) -> np.ndarray:
    """Random sub-window of the sequence, padded back to the original length.

    A pen stroke's start/end often contain little useful signal (pen
    approaching/leaving the paper, contact bounce). Random cropping exposes
    the model to sub-windows. If the crop is shorter than the original, the
    truncated tail is filled by repeating the edge value so the output shape
    is unchanged and downstream padding/truncation still works.
    """
    t = seq.shape[0]
    lo = int(t * (1.0 - min_frac) / 2)
    if lo < 1:
        return seq
    start = rng.integers(0, 2 * lo + 1)
    end = t - (2 * lo - start)
    crop = seq[start:end]
    if crop.shape[0] == t:
        return crop
    # pad back to length t with edge values (mirror would create discontinuities)
    pad = np.pad(crop, ((0, t - crop.shape[0]), (0, 0)), mode="edge")
    return pad


# --------------------------------------------------------------------------- #
# Augmentation policy
# --------------------------------------------------------------------------- #
@dataclass
class AugmentationConfig:
    """Knobs for the augmentation policy.

    The **defaults reproduce the legacy policy** - jitter, per-channel scale,
    magnitude warp, time warp - with the exact sigmas that produced the
    measured 64.8% -> 71.6% writer-independent jump on the bundled OnHW
    subset (augment x4, 2x BiLSTM-100; see the README's "Improving accuracy"
    section).

    The three IMU-specific transforms added later (rotation, channel dropout,
    random crop) are **off by default**, because turning them on silently
    would change what ``--augment N`` means and make that 71.6% figure
    non-reproducible. They have not been measured on this subset. Opt in with
    ``AugmentationConfig.extended()`` or ``--aug-policy extended``, and record
    whatever number the run produces before recommending them.
    """

    jitter_sigma: float = 0.05
    scale_sigma: float = 0.08
    mag_warp_sigma: float = 0.15
    mag_warp_knots: int = 4
    time_warp_sigma: float = 0.2
    time_warp_knots: int = 4
    rotation_max_deg: float = 10.0  # magnitude only; p_rotation gates it
    channel_dropout_p: float = 0.05
    crop_min_frac: float = 0.85
    # Probabilities of applying each transform (independent Bernoulli draws).
    # The last three default to 0 = the legacy policy.
    p_jitter: float = 1.0
    p_scale: float = 1.0
    p_mag_warp: float = 0.7
    p_time_warp: float = 0.7
    p_rotation: float = 0.0
    p_channel_dropout: float = 0.0
    p_crop: float = 0.0

    @classmethod
    def legacy(cls) -> "AugmentationConfig":
        """The measured policy: jitter + scale + magnitude warp + time warp."""
        return cls()

    @classmethod
    def extended(cls) -> "AugmentationConfig":
        """Legacy policy plus rotation, channel dropout and random crop.

        Unmeasured on the bundled subset - benchmark it before quoting a
        number for it.
        """
        return cls(p_rotation=0.5, p_channel_dropout=0.3, p_crop=0.3)


#: Policy name -> constructor, for the ``--aug-policy`` CLI flag.
AUG_POLICIES = {
    "legacy": AugmentationConfig.legacy,
    "extended": AugmentationConfig.extended,
}


def augment_one(
    seq: np.ndarray, rng: np.random.Generator, cfg: Optional[AugmentationConfig] = None
) -> np.ndarray:
    """Apply the default augmentation policy to one (T, C) sequence.

    The policy is the legacy jitter+scale+mag_warp+time_warp policy, plus
    three new transforms that are physically meaningful for IMU data:

    - random_rotation  - small 3D rotation of each Acc/Gyro/Mag triad
                         (pen-grip variation; preserves vector magnitudes)
    - channel_dropout  - zero out a channel for the whole sample
                         (sensor dropout failure mode)
    - random_crop      - random sub-window of the stroke
                         (start/end often contain little signal)

    The transform order is: rotations first (frame change), then magnitude
    transforms (gain drift), then time transforms (speed variation), then
    noise. This is the standard order from the time-series augmentation
    literature.
    """
    cfg = cfg or AugmentationConfig()
    s = seq.astype(np.float32, copy=True)

    if cfg.rotation_max_deg > 0 and rng.random() < cfg.p_rotation:
        s = random_rotation(s, rng, cfg.rotation_max_deg)
    if rng.random() < cfg.p_scale:
        s = per_channel_scale(s, rng, cfg.scale_sigma)
    if rng.random() < cfg.p_jitter:
        s = jitter(s, rng, cfg.jitter_sigma)
    if rng.random() < cfg.p_mag_warp:
        s = mag_warp(s, rng, cfg.mag_warp_sigma, cfg.mag_warp_knots)
    if rng.random() < cfg.p_time_warp:
        s = time_warp(s, rng, cfg.time_warp_sigma, cfg.time_warp_knots)
    if cfg.channel_dropout_p > 0 and rng.random() < cfg.p_channel_dropout:
        s = channel_dropout(s, rng, cfg.channel_dropout_p)
    if rng.random() < cfg.p_crop:
        s = random_crop(s, rng, cfg.crop_min_frac)
    return s


def augment_training(
    x: List[np.ndarray],
    y: np.ndarray,
    writers: np.ndarray,
    train_idx: np.ndarray,
    n_aug: int,
    seed: int,
    cfg: Optional[AugmentationConfig] = None,
) -> tuple:
    """Append ``n_aug`` augmented copies of each training sample.

    Mirrors the contract of the original ``onhw_models.augment_training``:
    original samples keep their indices, augmented copies are appended, so
    the val/test index arrays remain valid and never see augmented data.
    """
    rng = np.random.default_rng(seed)
    x = list(x)
    new_y, new_w, new_idx = [], [], []
    for j in train_idx:
        for _ in range(n_aug):
            x.append(augment_one(x[j], rng, cfg))
            new_idx.append(len(x) - 1)
            new_y.append(y[j])
            new_w.append(writers[j])
    if new_y:
        y = np.concatenate([y, np.array(new_y, dtype=y.dtype)])
        writers = np.concatenate([writers, np.array(new_w, dtype=writers.dtype)])
        train_idx = np.concatenate(
            [train_idx, np.array(new_idx, dtype=train_idx.dtype)]
        )
    return x, y, writers, train_idx


# Backwards-compatible aliases for the old private names used by tests.
_legacy_time_warp = time_warp
_legacy_mag_warp = mag_warp
