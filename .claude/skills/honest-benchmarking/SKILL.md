---
name: honest-benchmarking
description: Use when running, reporting, or reviewing any accuracy/CER/WER experiment in this repo - training runs, benchmark tables, README result updates, PR descriptions quoting a number, or adding a dataset loader. Use when asked to "improve accuracy", "benchmark", "compare configs", "why is accuracy low", or when about to write a number into a doc.
---

# Honest benchmarking

This repo exists because `legacy/cnn_gnn.py` reported ~99% and the honest held-out
figure was ~43-47%. Every rule here comes from a mistake actually made in this
codebase, most of them in PR #8.

## Before you run anything

Decide what would change your mind. A run that cannot come out badly is not an
experiment. Write down the number you expect and what you would conclude if it
came out two points lower.

## Seeding and noise

`--seed` alone does not pin a run. `tf.random.set_seed` does not reach the
Keras layer initialisers, so two runs at the same seed start from different
weights. Seeding goes through `keras.utils.set_random_seed` in
`imu2text/models.py`; a new entry point must do the same.

For any before/after comparison, pass `--deterministic`. It adds op
determinism and single-threaded execution, costs 30-50% wall-clock, and makes
the run bit-reproducible.

**Measure the noise floor on the dataset you are using, not another one.**
Same config, same seed, run twice:

- OnHW-chars_L (1,334 train / 416 test): about 5 points of spread
- Official OnHW-chars (19,819 / 7,956): about 0.2 points

A gain smaller than that spread is not a result. This is the single most
common way to fool yourself here, because the small dataset runs fast and
invites iteration.

## Reporting a number

Never write an accuracy without all four of:

1. **The split** - writer-independent or random
2. **Where the split came from** - published (`--onhw-chars` uses the official
   folds) or constructed here (`--split writer`)
3. **The dataset and its size** - the bundled subset, OnHW-chars_L and the
   official OnHW-chars are three different things and their numbers are not
   interchangeable
4. **The class count** - 52-class and 26-class differ by about 12 points for
   reasons that have nothing to do with the model

Single seed on one fold is a data point, not a result. Say so. The 30
published folds exist to be averaged.

## Read the train column

Train accuracy is where the diagnosis lives. On the official split:

| | Train | Test |
|---|--:|--:|
| 1xBiLSTM-64 | 90.1 | 69.2 |
| 2xBiLSTM-100 | 95.5 | 69.6 |
| 2x100 + every regulariser + augmentation | 99.2 | 70.4 |
| 1x64 + attention + augmentation + smoothing + schedule | 92.1 | 72.5 |

Capacity bought +0.4 test for +5 train. Piling regularisers onto that capacity
was worse than leaving augmentation off, because a model with enough
parameters memorises the augmented copies too. The best configuration has the
lowest train accuracy. When train is at 99% and test is stuck, more model is
the wrong instrument.

## Diagnose before optimising

When accuracy stops moving, count the errors. `--error-analysis` prints
confusion pairs and case-insensitive accuracy.

The known result on 52-class OnHW-chars: 38.4% of errors are a letter confused
with its own other case, all twelve top confusions are case pairs, and for
same-shape pairs (C/c, O/o, S/s, U/u, V/v, W/w, X/x, Z/z, K/k, P/p) the cue is
close to absent from the signal. Acceleration RMS separates them at AUC 0.541,
duration at 0.590. Acceleration goes as size over time squared and writers
form capitals both larger and proportionally faster, so the effects cancel.

That is a sensing limit. 26-class scoring, word context, or a position-sensing
modality moves it. Architecture does not.

## Don't judge a lever in isolation

Attention pooling scored 69.0 against a 69.2 baseline alone. With augmentation
it was worth +0.9, and +3.0 once label smoothing and the LR schedule joined.
Levers interact; a single-configuration null is weak evidence for dropping one.

## Dataset loaders

A synthetic fixture written alongside a loader encodes the same assumptions as
the loader and tests nothing about the real format. In PR #8 all 65 tests
passed while four loaders could not open the published archives.

Every loader needs an integration test against the real archive, skipped by
default. See `tests/test_real_data.py`, gated on `ONHW_DATA_DIR`.

Formats guessed wrong once, and worth checking first: labels as strings rather
than ints, fold directories named `0` rather than `fold_0`, labels
right-padded with the blank index, split files under different names from
their unsplit equivalents. Real archives also contain degenerate records -
three OnHW-chars samples have zero timesteps.

Never fill missing metadata with a plausible value. Absent writer IDs became
zeros once, which reads as "one writer" and silently turns a
writer-independent split into a leaky one. Use a sentinel and make consumers
reject it.

## Leakage checklist

- Normalization fitted on train indices only, unless the mode is documented as
  transductive (`--norm per_writer` is, and any number from it must say so)
- Augmentation applied to train only, never to val or test
- Val carved from the training half, never from the published test half
- If an archive ships a split, use it rather than re-deriving one. The symbols
  `dep` archive shares all 27 writers across train and val on purpose;
  synthesising a writer-disjoint split from it reports a writer-independent
  number from writer-dependent data.
