# Why signal filtering hurt accuracy

Both filters added in `imu2text/filters.py` lowered writer-independent
accuracy on OnHW-chars `both/indep/fold0`:

| Filter | Test % | Delta |
|---|--:|--:|
| none | 67.32 | - |
| lowpass, 15 Hz | 66.39 | -0.93 |
| orientation (Madgwick) | 63.76 | -3.56 |

The two results have different causes. One is a property of the data. The
other was a defect in the filter, and the number above does not measure what
it claimed to.

## Low-pass: the filter removed signal, not noise

The premise was that an IMU at 100 Hz carries sensor noise well above the
bandwidth of hand motion, so a 15 Hz cutoff should discard noise and keep the
writing. Measuring the spectrum says otherwise.

Power above the cutoff, averaged over 3,000 sequences:

| Channels | Above 15 Hz | Above 25 Hz |
|---|--:|--:|
| Accelerometers | **53.9%** | 35.8% |
| Gyroscope | 4.8% | 1.9% |
| Force | 3.2% | 1.4% |

More than half the accelerometer energy sits above the cutoff. That alone
does not prove it is signal, so the next question is whether the discarded
band predicts the class. Splitting each sequence into its low and high bands
and fitting a logistic regression on per-channel summary statistics:

| Band | Accuracy | Chance |
|---|--:|--:|
| Below 15 Hz | 9.22% | 1.92% |
| Above 15 Hz | **7.11%** | 1.92% |

The high band alone predicts the character at 3.7 times chance from 26 crude
features. It is not noise. Pen-tip friction against paper, the impact of a
pen-down, and the sharp reversals at stroke corners are all fast events, and
they are exactly the events that distinguish one letterform from another.

This is the expected outcome once stated plainly: the CNN trunk already
learns its own filters, and a fixed cutoff can only remove what the model
might otherwise have used. **-0.93 is the cost of discarding half the
accelerometer bandwidth**, and the model recovers most of it because the low
band is also informative.

## Orientation: the measurement was invalid

The premise was better. Writers hold a pen at different angles, which rotates
the sensor frame and appears as a per-writer nuisance in every channel.
Estimating the orientation with a Madgwick filter and re-expressing
acceleration in a fixed earth frame should cancel it.

The -3.56 does not test that premise. The filter had a units defect.

**The archives store raw sensor counts, not physical units.** The filter
converted the gyroscope with `np.deg2rad(gyro)`, treating a reading of 1,917
as 1,917 deg/s. The observed 99th-percentile gyroscope value is 1,917 counts,
so the integrator was fed up to 33.5 rad/s, which is a rotation of about 0.33
radians - 19 degrees - **per 10 ms timestep**.

The symptom was visible before the cause. Tracking how far the quaternion
moves per step, a converging filter settles; this one did not:

| | step 0 | step 4 | step 9 | step 19 | step 38 |
|---|--:|--:|--:|--:|--:|
| As deg/s (the defect) | 0.0519 | 0.0660 | 0.0760 | 0.0756 | 0.0762 |
| Raw counts, corrected | 0.0032 | - | - | - | 0.0048 |

The per-step change *grew* and never dropped below half its initial rate
inside a sequence, which average 92 samples. The orientation estimate was
spinning, so the rotation applied to the accelerometers was arbitrary. That
scrambles the channels, and -3.56 is the cost of scrambling them.

One hypothesis was checked and did **not** hold: that removing the gravity
term discards discriminative tilt information. The steady component of the
acceleration vector predicts the class at 2.56% against 1.92% chance, which
is barely above chance. Loss of tilt information is not what caused the drop.

A second check confirmed Madgwick's core assumption is otherwise sound here.
The filter assumes the accelerometer reads gravity plus a small perturbation.
The ratio of the moving component to the steady one has a median of 0.13, and
in no sampled sequence did the motion exceed the steady term. Gravity does
dominate; the gyroscope input was the problem.

### Establishing the real units

The archive settles the first half of the question. `readme.txt` inside
`onhw-chars_2021-06-30` states:

> The OnHW-chars dataset does not contain or consider any sensor calibration.

So the values are raw counts, confirmed by the publisher rather than inferred.
The pen is a STABILO DigiPen whose front accelerometer and gyroscope are an
STM LSM6DSL (Ott et al., IMWUT 2020), but neither the paper nor the readme
records the configured full-scale ranges.

**The accelerometer scale can be derived, because gravity is a known
quantity.** The median magnitude of the front accelerometer is 16,550 counts,
and the p10-p90 range is 16,313 to 17,050. Against the LSM6DSL's options:

| Full scale | LSB/g | Median reads |
|---|--:|--:|
| **+/-2 g** | **16,384** | **1.010 g** |
| +/-4 g | 8,192 | 2.020 g |
| +/-8 g | 4,096 | 4.040 g |
| +/-16 g | 2,048 | 8.081 g |

A pen spends most of its time reading 1 g, so the range is +/-2 g. That is a
derivation, not a guess.

**The gyroscope scale has to be inferred**, since no comparable constant is
available. The signal never approaches the int16 limit (max 18,987 of 32,767),
so it is not saturating and the range cannot be read off a clipped signal.
Integrating each candidate range over a character gives:

| Full scale | LSB/dps | Peak rate | Total swing per character |
|---|--:|--:|--:|
| +/-250 dps | 114.3 | 16.7 dps | 4.4 deg |
| +/-500 dps | 57.1 | 33.5 dps | 8.7 deg |
| +/-1000 dps | 28.6 | 66.8 dps | 17.4 deg |
| **+/-2000 dps** | **14.3** | **133.6 dps** | **34.9 deg** |

4.4 degrees of total pen rotation while forming a letter is not credible;
about 35 is. `GYRO_UNITS_PER_DPS` is therefore 14.3, flagged in the source as
inferred rather than documented, and `--gyro-scale` overrides it.

## What to take from this

- **A filter that silently does nothing looks exactly like a filter that does
  not help.** The first version of this experiment returned three identical
  accuracies to the decimal because `apply_filter` was never called, and that
  was read as a result rather than a bug. `tests/test_pipeline.py` now pins
  that each filter changes the data.
- **Units are part of the interface.** These archives ship raw counts with no
  datasheet. Any code that assumes physical units needs to say so where it
  makes the assumption, and preferably check it.
- **A filter that fails to converge is visible without knowing why.** The
  per-step quaternion change was diagnostic on its own, before the cause was
  found. Physics-based preprocessing should report whether it converged.
- Removing bandwidth from an accelerometer is not free on this task. Over
  half the energy, and independently predictive content, lives above 15 Hz.
