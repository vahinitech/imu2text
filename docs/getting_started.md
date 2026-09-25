# Getting started

What this project does, where the accuracy stands, how to run it and what to
work on next. If a term is unfamiliar, see the [glossary](glossary.md).

## The problem

Someone writes on ordinary paper with a ballpoint pen that has sensors inside.
From the sensor stream alone, work out what they wrote.

The hard part: **the pen does not know where it is.** Its accelerometers and
gyroscope measure motion, not position. Getting position from acceleration
means integrating twice, and the error drifts within a single character. So
the pen cannot redraw the letter, and image-based handwriting recognition does
not apply.

What you get instead is a 13-channel time series at 100 Hz:

| Channels | Sensor |
|---|---|
| 0-2 | front accelerometer, x/y/z |
| 3-5 | rear accelerometer, x/y/z |
| 6-8 | gyroscope, x/y/z |
| 9-11 | magnetometer, x/y/z |
| 12 | pen-tip force |

One character is roughly 50 timesteps. The task is to map a variable-length
13-channel signal to a label.

Tablets already recognise handwriting well. A pen matters because it leaves the
act of writing alone. A child learning letterforms needs the friction of paper
and the weight of a real pen, and a glass screen teaches a different motion. A
classroom of pens is stationery, not a procurement project. Homework, exams and
forms are already on paper. The application behind this repo is handwriting
assessment in schools: read what a student wrote, and later say something
useful about how they wrote it.

## Where the accuracy is

On the official OnHW-chars benchmark (52 classes, A-Z and a-z,
writer-independent `both/indep/fold0`, so every test writer is unseen), the
best model here scores **72.5%** against 68.06% for CNN+BiLSTM in Ott et al.,
ACM MM 2022, Table 3. Single seed, fold 0, CPU only. The full tables, the
ablations and the other datasets are in [benchmarks.md](benchmarks.md).

72.5% sounds low if you are used to MNIST. For 52 classes, an unseen writer and
a sensor that never sees the letter's shape, it is not.

## The model

![Architecture](../results/architecture.png)

1. **CNN trunk.** Two Conv1D layers with batch norm and max pooling read short
   local patterns (a change in stroke direction, a pen lift) and downsample
   time by 4, so a character goes from 100 timesteps to 25.
2. **BiLSTM.** Reads those 25 steps forwards and backwards. The difference
   between `c` and `a` is what happens after the curve, so the backward pass
   matters.
3. **Read-out.** The baseline takes the BiLSTM's final state. The attention
   variant keeps all 25 steps, learns a weight for each, and takes the weighted
   average plus a max.

Roughly 145k parameters for the baseline, 158k with attention. Keeping it small
is deliberate: extra capacity went into memorising the training set, not into
test accuracy (see [benchmarks.md](benchmarks.md)).

Sequence tasks (words, equations) use the same trunk with a CTC head, in
`imu2text/seq2seq.py`.

## Why characters get confused

Read this before trying to improve anything. 43% of the remaining errors are a
letter confused with its own other case (`s`→`S`, `o`→`O`, `w`→`W`), and all
twelve of the commonest confusions are case pairs. Fold case away and the same
model scores 84.3% instead of 72.5%.

For about ten letters (C/c, O/o, S/s, U/u, V/v, W/w, X/x, Z/z, K/k, P/p) the two
cases are the same shape at a different size. Writers form capitals larger and
proportionally faster, so the size difference cancels out of the acceleration
signal. This is a sensing limit: no architecture recovers what the sensor never
recorded. The measurements are in the error-analysis section of
[benchmarks.md](benchmarks.md).

## What to do next

Ordered by expected value per unit of work. Each has an issue.

**Give the model context** ([#11](https://github.com/vahinitech/imu2text/issues/11)).
Probably the biggest gain left, and it needs no new sensor. In real writing,
case is decided by position in a word: `cat` and `Cat` differ by where the
letter sits, not its shape. A word-level CTC model with a lexicon gets case
almost for free. The lexicon decoder in `imu2text/words.py` has been run on
OnHW-Words500 (results in [benchmarks](benchmarks.md)) but not yet used to
settle case for character recognition.

**Know when the model is guessing** ([#13](https://github.com/vahinitech/imu2text/issues/13)).
The cheapest thread: no download, no new architecture. A 72.5% recogniser that
can flag its own uncertain 28% is usable in a classroom; one that cannot is a
demo. First check whether the case confusions are confidently wrong. If they
are already low-confidence, deferring on them recovers most of that 12-point
penalty in practice.

**Average over the folds** ([#9](https://github.com/vahinitech/imu2text/issues/9)).
Every number above is one seed on one fold. The 30 split directories are 3
case settings × 2 protocols × 5 folds, so each task (for example `both/indep`)
is averaged over its own 5 folds. A few hours of CPU.

**Split letter identity from case** ([#10](https://github.com/vahinitech/imu2text/issues/10)).
A 26-way head plus a binary case head matches the diagnosis and lets the case
decision be calibrated or deferred on its own. It could also come out below the
current model if the two heads are independent.

**Hybrid classical + deep** ([#12](https://github.com/vahinitech/imu2text/issues/12)).
Predicted at 0 to +2 points, because it works on the 57% of errors that are not
case. A null result closes the direction cheaply.

More in [roadmap.md](roadmap.md).

## Other languages and scripts

Everything here is Latin script, from German and English data. The sensor, the
13 channels, the CNN+BiLSTM trunk, CTC, augmentation and the evaluation harness
assume nothing about the alphabet. `imu2text/models.py` infers the class set
from the labels, so classification needs no code change; the seq2seq charset is
a constant per dataset.

What does change is the learning problem. Devanagari and Telugu have far more
glyph units than 52, and Telugu composes consonant-vowel clusters. Scripts
written right to left, or with conjuncts formed in several passes, produce a
different signal, and Latin results do not predict how they will go. Where
characters connect (Arabic, Devanagari's shirorekha) segmentation gets harder,
which pushes toward the sequence model.

Two smaller IMU datasets exist:

- Sharma et al., "Dataset of inertial measurements for writing Punjabi
  characters using IMU sensors" (Data in Brief, 2024, Akal University
  Bathinda). Gurmukhi script, collected across Punjabi writers.
- Gupta and Mishra, "A Dataset of Inertial Measurement Units for Handwritten
  English Alphabets" (IIT BHU Varanasi). Collected in India, English alphabet.

Neither reaches OnHW's 119 writers, and for writer-independent evaluation the
writer count matters more than the sample count. A new collection needs at
least a few dozen writers, with writer identity recorded so whole writers can be
held out. The sensible order is word-level recognition on the Latin data first,
then collection, then transfer.

## Install and run

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m imu2text.seq2seq --demo          # verifies the pipeline, no download
pytest                                     # no dataset needed
```

Then download the benchmark and reproduce the headline number:

```bash
python -m imu2text.download onhw_chars --out ./data       # 896 MB, once
python -m imu2text.models --models cnn_bilstm_attn \
    --onhw-chars data/onhw-chars_2021-06-30 \
    --case both --dependency indep --fold 0 \
    --augment 2 --aug-policy extended \
    --label-smoothing 0.1 --lr-schedule --epochs 30 --error-analysis
```

About 15 minutes on 4 CPU cores. `--error-analysis` prints the confusion
breakdown, which is where to look when a change does not help. The other
archives and loaders are in [datasets.md](datasets.md).

Results and figures live in `results/`; the numbers behind them are written up
in [benchmarks.md](benchmarks.md).

## Contributing

Two rules that save time:

1. **Measure the noise floor before believing an improvement.** Run the same
   config at the same seed twice. On the official split the spread is about 0.2
   points; on the small OnHW-chars_L set it is about 5. A gain smaller than the
   spread is not a result. Use `--deterministic` for comparisons.
2. **Test loaders against the real archives.** A synthetic fixture written next
   to the loader says nothing about the real data format: four loaders here
   once passed 65 tests while unable to open the published archives. Real-data
   tests live in `tests/test_real_data.py`, gated on `ONHW_DATA_DIR`.

The working rules for changes are in [CLAUDE.md](../CLAUDE.md), and the
benchmarking conventions in `.claude/skills/benchmarking/SKILL.md`.
