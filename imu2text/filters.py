"""Signal conditioning for IMU sequences: low-pass and orientation removal.

Two questions this module exists to answer, both by measurement rather than
argument:

1. Does denoising help? An IMU at 100 Hz carries high-frequency sensor noise
   well above the bandwidth of hand motion. A low-pass filter removes it.
   Against that, the CNN trunk already learns its own filters, so a fixed
   filter may only remove signal the model would have used.

2. Does removing pen orientation help? Writers hold a pen at different
   angles, which rotates the sensor frame and shows up as a per-writer
   nuisance in every channel. That is the same variation ``augment.
   random_rotation`` simulates. An orientation filter estimates the rotation
   and undoes it, expressing acceleration in a fixed earth frame instead of
   the pen's own. If grip variation is a real part of the writer-independent
   gap, this should close some of it.

   The counter-argument is that pen tilt is not purely nuisance. How a writer
   holds the pen correlates with how they form letters, and gravity's
   direction in the sensor frame encodes tilt. Removing it may discard signal.

``madgwick_orientation`` implements Madgwick's complementary filter from the
published equations (S. Madgwick, "An efficient orientation filter for
inertial and inertial/magnetic sensor arrays", 2010). The algorithm is
standard and widely reimplemented; this is written from the paper's
formulation, not adapted from an existing implementation.

Channel layout is the OnHW one, so the accelerometer and gyroscope triads are
at fixed offsets. See ``imu2text.augment.SENSOR_GROUPS``.
"""

from __future__ import annotations

from typing import List

import numpy as np

ACC1 = [0, 1, 2]
ACC2 = [3, 4, 5]
GYRO = [6, 7, 8]

SAMPLE_RATE_HZ = 100.0


def lowpass(
    seq: np.ndarray, cutoff_hz: float = 15.0, sample_rate: float = SAMPLE_RATE_HZ
) -> np.ndarray:
    """One-pole low-pass, applied forwards then backwards for zero phase lag.

    Running it in both directions matters here: a causal filter delays the
    signal, and the model reads timing, so a phase shift would be a change of
    content rather than a removal of noise.
    """
    if seq.shape[0] < 3:
        return seq
    dt = 1.0 / sample_rate
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    alpha = dt / (rc + dt)

    def sweep(x):
        out = np.empty_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
        return out

    forward = sweep(seq.astype(np.float64))
    return sweep(forward[::-1])[::-1].astype(np.float32)


def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def madgwick_orientation(
    acc: np.ndarray,
    gyro_dps: np.ndarray,
    beta: float = 0.1,
    sample_rate: float = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Per-timestep orientation quaternion from accelerometer and gyroscope.

    The gyroscope integrates to an orientation that drifts; the accelerometer
    sees gravity and so fixes the two tilt axes but not heading. Madgwick's
    filter blends them: integrate the gyro, then correct along the gradient of
    the error between the measured gravity direction and the one the current
    orientation predicts. ``beta`` sets how hard the accelerometer pulls.

    Returns an (T, 4) array of unit quaternions in w, x, y, z order.
    """
    n = len(acc)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    out = np.empty((n, 4))
    dt = 1.0 / sample_rate
    gyro = np.deg2rad(np.asarray(gyro_dps, dtype=np.float64))

    for t in range(n):
        gx, gy, gz = gyro[t]
        qw, qx, qy, qz = q
        # Quaternion derivative from the angular rate.
        q_dot = 0.5 * np.array(
            [
                -qx * gx - qy * gy - qz * gz,
                qw * gx + qy * gz - qz * gy,
                qw * gy - qx * gz + qz * gx,
                qw * gz + qx * gy - qy * gx,
            ]
        )

        a = np.asarray(acc[t], dtype=np.float64)
        if np.linalg.norm(a) > 1e-9:
            ax, ay, az = _normalise(a)
            # Error between measured gravity and the direction this
            # orientation predicts, and its gradient in quaternion space.
            f = np.array(
                [
                    2.0 * (qx * qz - qw * qy) - ax,
                    2.0 * (qw * qx + qy * qz) - ay,
                    2.0 * (0.5 - qx * qx - qy * qy) - az,
                ]
            )
            j = np.array(
                [
                    [-2.0 * qy, 2.0 * qz, -2.0 * qw, 2.0 * qx],
                    [2.0 * qx, 2.0 * qw, 2.0 * qz, 2.0 * qy],
                    [0.0, -4.0 * qx, -4.0 * qy, 0.0],
                ]
            )
            step = _normalise(j.T @ f)
            q_dot = q_dot - beta * step

        q = _normalise(q + q_dot * dt)
        out[t] = q
    return out


def _rotate_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate (T, 3) vectors from the sensor frame into the earth frame."""
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # Rows of the rotation matrix, built per timestep.
    r00 = 1 - 2 * (qy * qy + qz * qz)
    r01 = 2 * (qx * qy - qw * qz)
    r02 = 2 * (qx * qz + qw * qy)
    r10 = 2 * (qx * qy + qw * qz)
    r11 = 1 - 2 * (qx * qx + qz * qz)
    r12 = 2 * (qy * qz - qw * qx)
    r20 = 2 * (qx * qz - qw * qy)
    r21 = 2 * (qy * qz + qw * qx)
    r22 = 1 - 2 * (qx * qx + qy * qy)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return np.stack(
        [
            r00 * x + r01 * y + r02 * z,
            r10 * x + r11 * y + r12 * z,
            r20 * x + r21 * y + r22 * z,
        ],
        axis=1,
    )


def orientation_normalise(
    seq: np.ndarray, beta: float = 0.1, sample_rate: float = SAMPLE_RATE_HZ
) -> np.ndarray:
    """Express both accelerometer triads in a fixed earth frame, gravity removed.

    The gyroscope channels are left alone: angular rate about the pen's own
    axes is what it is, and rotating it into the earth frame would not make it
    more comparable across writers.

    Returns an array of the same shape, so it drops into the pipeline
    wherever a raw sequence would go.
    """
    if seq.shape[0] < 2 or seq.shape[1] <= max(GYRO):
        return seq
    out = seq.astype(np.float32, copy=True)
    gyro = seq[:, GYRO]
    for cols in (ACC1, ACC2):
        acc = seq[:, cols]
        q = madgwick_orientation(acc, gyro, beta=beta, sample_rate=sample_rate)
        earth = _rotate_by_quaternion(np.asarray(acc, dtype=np.float64), q)
        # In the earth frame gravity sits on one axis; subtracting the median
        # leaves linear acceleration without assuming the units of g.
        earth = earth - np.median(earth, axis=0, keepdims=True)
        out[:, cols] = earth.astype(np.float32)
    return out


FILTERS = {
    "none": lambda seq: seq,
    "lowpass": lowpass,
    "orientation": orientation_normalise,
}


def apply_filter(x: List[np.ndarray], name: str) -> List[np.ndarray]:
    """Run one of the named filters over a list of (T, C) sequences."""
    if name not in FILTERS:
        raise ValueError(f"unknown filter: {name!r}; pick from {sorted(FILTERS)}")
    if name == "none":
        return x
    fn = FILTERS[name]
    return [fn(np.asarray(s, dtype=np.float32)) for s in x]
