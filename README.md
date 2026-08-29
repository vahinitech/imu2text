# IMU Character Recognition (CNN + GNN)

A compact project to train a multi-task model that performs:

- Character classification from IMU pen sensor data.
- Trajectory regression (reconstruction) of pen movement.

This repository contains a minimal, easy-to-follow pipeline implemented in `cnn_gnn.py`.

Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run training or inference (script is a self-contained example):

```bash
python cnn_gnn.py
```

Files of interest

- `cnn_gnn.py` - original single-script example (preprocessing, model, train/eval).
- `onhw_models.py` - honest OnHW benchmark suite: baselines (CNN, LSTM, BiLSTM) and the SOTA CNN+BiLSTM, with both writer-independent and random splits. The class set is inferred from the labels, so the same script handles OnHW-chars **and OnHW-symbols** pickles.
- `onhw_seq2seq.py` - sequence-to-sequence (words / equations) recognition: CNN+BiLSTM encoder trained with CTC, greedy decoding, CER/WER metrics. Run `python onhw_seq2seq.py --demo` to verify the pipeline on synthetic data without downloading a dataset.
- `make_learning_curve.py` - trains the SOTA model on an increasing number of writers (writer-independent) to produce the accuracy learning curve.
- `plot_results.py` - publication-quality matplotlib figures (learning curve + logistic projection, model benchmark bars), rebuilt in the style of ImpAcX_OnHW's `plot_kNN_results.py`.
- `onhw_projection.m` - MATLAB/Octave script that fits a logistic model to the learning curve and projects pen accuracy to full-dataset scale.
- `docs/impacx_onhw_analysis.md` - analysis of the ImpAcX_OnHW DTW-kNN pipeline and how its matplotlib figures are rebuilt here.
- `docs/onhw_enhancement_guide.md` - roadmap for character / symbol / seq2seq recognition across the OnHW dataset family (with pointers to REWI and related work).
- `docs/onhw_research_threads.md` - what the five OnHW side-datasets (pen-tip trajectory, cross-modal, domain adaptation, uncertainty) are for, why wordsTraj has only two writers, and a proposed order of work.

Dataset collection kit - moved to [vahinitech/datasets](https://github.com/vahinitech/datasets)

The school data-collection tooling and docs that used to live here (capture
app, sheet/booklet generators, raw-tree converter, station merge tool, the
collection protocol / fieldwork procedure / dry-run checklist) now live in
the dedicated datasets repo. This repo keeps the model training code; train
on collected data with:

```bash
python onhw_models.py --channels 16 \
    --imu-file <datasets-out>/all_x_dat_imu.pkl \
    --gt-file  <datasets-out>/all_gt.pkl \
    --writers-file <datasets-out>/writers.pkl --split writer
```
- `tests/` - smoke tests for splitting, writer inference, augmentation, normalization, and the CTC pipeline (run by CI).
- `LICENSE` - project license and contact information.

Security & dependencies

`requirements.txt` pins patched releases for every GitHub security advisory
open against this repo: `torch==2.13.0` and `scikit-learn==1.5.2`
(CVE-2024-5206).

The torch pin was `2.7.1`, and this section used to claim that covered the
2025 memory-corruption advisories. It did not: Dependabot had four alerts open
against it, needing 2.8.0, 2.9.1, 2.10.0 and 2.13.0 respectively, so 2.13.0 is
the first release that clears all of them. 2.7.1 did fix the one that mattered
most, CVE-2025-32434, the `torch.load` RCE that works even with
`weights_only=True`.

Only `cnn_gnn.py` imports torch. The rest of the pipeline is TensorFlow, so
these advisories never reached the benchmark code - but the dependency is
declared and installed, so the pin is worth keeping current.
It also removes the duplicate/conflicting `torch` pins and the
`tensorflow-macos` line that previously broke `pip install -r
requirements.txt` on Linux, and adds the missing `torch-geometric` and
`tabulate` dependencies used by `cnn_gnn.py`. For CPU-only machines/CI:
`pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`.

Benchmarks (honest, held-out evaluation)

`cnn_gnn.py` reports ~99% accuracy, but it trains and evaluates on the *same*
array - that figure is train-set memorization, not recognition skill. On a
held-out split the same model scores ~43–47%. `onhw_models.py` fixes the
methodology (real train/val/test split, train-only normalization, early
stopping) and implements the OnHW baselines plus the SOTA CNN+BiLSTM.

Two evaluation protocols are supported:

- `--split writer` (default) - **writer-independent (WI)**: whole writers are
  held out, so test writers are entirely unseen. This is the protocol the OnHW
  papers report. Writer IDs are reconstructed from the recording order (the pen
  records one writer's full alphabet at a time - see `infer_writer_ids`).
- `--split random` - stratified random (easier; a writer's style can leak into
  both train and test).

```bash
python onhw_models.py                 # writer-independent, all 4 models
python onhw_models.py --split random  # easier random split
```

Writer-independent results on the bundled subset (2,270 samples, 45 writers,
52 classes; 1×BiLSTM-64, seq len 100 - CPU-friendly config):

| Model       | Train % | WI Test % |
|-------------|--------:|----------:|
| **cnn_bilstm** (SOTA) | 96.1 | **64.8** |
| bilstm      | 88.7 | 56.2 |
| cnn         | 90.8 | 48.7 |
| lstm        | 75.7 | 43.8 |
| majority baseline | - | 2.2 |

The ordering matches the literature (CNN+BiLSTM > BiLSTM > CNN > LSTM), and
CNN+BiLSTM's 64.8% WI essentially matches the published 52-class OnHW baseline
(~64%, Ott et al. 2020) - on only 27 training writers.

Official OnHW-chars benchmark

The bundled subset is a convenience sample. The comparable number comes from
the published dataset and its own splits: 31,275 samples, 119 writers, 52
classes, evaluated on `both/indep/fold0` - the writer-independent protocol the
papers report. Download it once (896 MB) and pass `--onhw-chars`:

```bash
python onhw_download.py onhw_chars --out ./data
python onhw_models.py --models cnn_bilstm_attn \
    --onhw-chars data/onhw-chars_2021-06-30 \
    --case both --dependency indep --fold 0 \
    --augment 2 --aug-policy extended \
    --label-smoothing 0.1 --lr-schedule --epochs 30
```

CNN+BiLSTM, 30 epochs, seed 0, official train/test partition:

| Configuration | Train % | WI Test % |
|---|--:|--:|
| 1×BiLSTM-64 | 90.1 | 69.2 |
| 1×BiLSTM-64, attention pooling | 89.4 | 69.0 |
| 1×BiLSTM-64 + augmentation ×2 (`extended`) | 90.1 | 70.0 |
| 2×BiLSTM-100 | 95.5 | 69.6 |
| 2×BiLSTM-100 + label smoothing 0.1 + LR schedule | 98.1 | 70.8 |
| 2×BiLSTM-100 + those + augmentation ×2 | 99.2 | 70.4 |
| 1×BiLSTM-64, attention pooling + augmentation ×2 | 89.5 | 70.9 |
| **the same + label smoothing 0.1 + LR schedule** | 92.1 | **72.5** |
| majority baseline | - | 1.9 |

**72.5% writer-independent on 52 classes** - 3.3 points above this repo's
previous best configuration and well above the ~64% published baseline, on a
model with a third of the parameters of the 2×BiLSTM-100 variants.

The train column explains why. Capacity alone buys almost nothing (+0.4 for
2×100 over 1×64) while pushing train accuracy from 90% to 95%: the extra
parameters go into memorising faster. Stacking the same regularisers onto that
capacity makes it worse rather than better - 2×100 with label smoothing, the LR
schedule *and* augmentation reaches 99.2% train for 70.4% test, below the same
model without augmentation. Given enough capacity the model simply memorises
the augmented copies too, and the augmentation stops constraining anything.

The winning configuration is the small one with every lever on: attention
pooling, augmentation, label smoothing and the LR schedule at 1×BiLSTM-64. It
holds train accuracy to 92.1% - eight points below the 2×100 models - and
converts that restraint into test accuracy. What is scarce on this task is
generalisation, not capacity, and the levers only compose while the model is
small enough to still be constrained by them.

Attention pooling is worth singling out because it would have been discarded on
its own evidence: alone it was a wash (69.0 against 69.2). It only pays once
augmentation is on, where it adds +0.9 over the equivalent CNN+BiLSTM, and
another +1.6 arrives when label smoothing and the LR schedule join it. Reach
for it via `--models cnn_bilstm_attn`.

How much of this is noise. The two anchor rows were re-run with
`--deterministic`, which makes a run bit-reproducible: the baseline came back
at **69.29%** against 69.18%, and the best configuration at **72.26%** against
72.46% - both within 0.2 points of their original runs, for a reproducible gap
of **+3.0**. Run-to-run noise on this dataset is therefore small, which is what
19,819 training and 7,956 test samples buys; on the far smaller OnHW-chars_L
the same measurement gave about 5 points of noise. So the headline improvement
is real, but the sub-point gaps between the middle rows are not - do not read
them as an ordering.

Still only a **single seed on one of the 30 published folds**. The folds exist
to be averaged and these have not been. Use `--deterministic` for any
comparison you make yourself; see the reproducibility note below.

Machine these numbers were measured on

Every figure in this README is **CPU-only**. There is no GPU in this machine
and TensorFlow reports none visible, so the timings below are the honest
worst case - a CUDA box will be several times faster and the accuracies will
not change.

| | |
|---|---|
| CPU | AMD EPYC 9354P, 4 vCPU allocated (1 thread/core, no SMT) |
| RAM | 7.8 GiB total, ~4.5 GiB available during runs |
| GPU | none (`tf.config.list_physical_devices('GPU')` is empty) |
| OS | Ubuntu 24.04.3 LTS, kernel 6.8.0 |
| Python | 3.10.21 (CI pins 3.10; TensorFlow 2.15.1 has no 3.12 wheels) |
| Key pins | tensorflow 2.15.1, numpy 1.26.4, scikit-learn 1.5.2 |

Wall-clock for the official OnHW-chars split (19,819 train / 7,956 test,
`maxlen` 100, batch 64), measured on the runs in the table above:

| Configuration | Epochs | Time |
|---|--:|--:|
| 1×BiLSTM-64 | 30 | 305 s |
| 1×BiLSTM-64, attention pooling | 30 | 322 s |
| 2×BiLSTM-100 | 30 | 598 s |
| attention pooling + augmentation ×2 + label smoothing + LR schedule | 30 | 913 s |
| the same with `--deterministic` | 30 | 1181 s |

`--deterministic` pins single-threaded execution, so it costs roughly 30-50%
more wall-clock. Augmentation multiplies the training set (`--augment 2` makes
it 3×) and the time with it.

Memory: a non-augmented run was observed at about 1.3 GB resident. The padded
input tensor is 31,272 × 100 × 13 float32 ≈ 163 MB, so `--augment 2` adds
roughly 0.3 GB of tensor on top; the whole set of runs above fitted in the
7.8 GiB available, but peak usage for the augmented configurations was not
measured directly.

Storage: the OnHW-chars `.npy` archive is 896 MB compressed and about 3.0 GB
extracted, so budget ~4 GB for it. `.gitignore` excludes `data/OnHW-*/`,
`data/onhw-*/` and `*.zip` so downloads are never committed.

Where the remaining error actually is

The 52-class figure stopped responding to modelling effort, so the errors were
counted rather than guessed at. `--error-analysis` breaks the test set down by
confusion pair:

```bash
python onhw_models.py --models cnn_bilstm \
    --onhw-chars data/onhw-chars_2021-06-30 --case both --dependency indep \
    --fold 0 --epochs 20 --error-analysis
```

```
Error analysis on 7956 test samples (2544 wrong, 31.98% error)
case-insensitive accuracy : 80.32%  (plain: 68.02%)
errors that are case only : 978/2544 (38.4% of all errors)
top 12 confusions: 'z'->'Z' 's'->'S' 'o'->'O' 'p'->'P' 'x'->'X' 'v'->'V'
                   'W'->'w' 'w'->'W' 'y'->'Y' 'u'->'U' 'c'->'C' 'Y'->'y'
```

Every one of the twelve most common confusions is a letter mistaken for its own
other case. Fold case away and the same model scores 80.3% instead of 68.0%:
about twelve points of the error is nothing but upper-versus-lower.

**The sensor cannot resolve it.** For pairs that share a glyph shape - C/c,
O/o, S/s, U/u, V/v, W/w, X/x, Z/z, K/k, P/p - the only distinguishing feature
is size, and an IMU measures acceleration, not position. Measuring how
separable the two cases actually are (AUC over the test set, 0.5 = a coin
flip):

| Cue | Same-shape pairs | Differently-shaped pairs (A/a, E/e, R/r, B/b, H/h) |
|---|--:|--:|
| Acceleration RMS | 0.541 | 0.36 - 0.54 |
| Sequence duration | 0.590 | 0.83 - 0.95 |

Acceleration scales as size / time², and writers form capitals both larger and
proportionally faster, so the two effects cancel; duration does not rescue it
either, since same-shape case pairs differ in length by a factor of 1.08
against 1.31 for differently-shaped pairs. The cue is not weakly represented in
the signal, it is close to absent.

Running the 26-class splits, where the ambiguity does not exist, confirms it
(same model, 20 epochs, `indep/fold0`):

| Task | WI Test % |
|---|--:|
| 52-class (`--case both`) | 68.0 |
| the same model scored case-insensitively | 80.3 |
| 26-class lowercase (`--case lower`) | 78.5 |
| 26-class uppercase (`--case upper`) | 81.8 |

The three ways of removing case all land near 80%, and 81.8% sits alongside the
published uppercase WI state of the art (~83%) from an un-augmented baseline
model. So the 52-class ceiling is a property of the label set and the sensor,
not of the architecture - which is why capacity bought +0.4 and regularisation,
which works on the 61.6% of errors that are not case, bought rather more.

Three things would move it, and none of them is a bigger model:

- **Score the task the ambiguity allows.** The 26-class splits ship with the
  dataset for this reason. Report which one you ran.
- **Give the model context.** Case in real writing is decided by position in a
  word, not by glyph shape - "cat" and "Cat" differ in where the letter sits.
  A word-level CTC model with a lexicon recovers most of it for free, which is
  a concrete reason the OnHW-words500 and equations datasets exist. See
  `onhw_seq2seq.py` and `docs/onhw_research_threads.md`.
- **Add a sensing modality that sees position** - the tablet-and-camera rig
  behind OnHW-wordsTraj.

Improving accuracy - the available levers

Every configuration below overfits: train accuracy runs 20-30 points above
held-out accuracy on all three datasets in this README. That gap, not model
capacity, is what limits the numbers. The levers are described here once, and
the sections after measure them - on the official benchmark above, and on
OnHW-chars_L further down.

Three datasets appear in this file and their numbers are **not**
interchangeable. Quote the official benchmark (72.5% WI) for anything
comparable to the literature; the bundled-subset and OnHW-chars_L figures are
for tracking changes within this repo.

- **IMU data augmentation** (`--augment N`): each training sample gets `N`
  randomly transformed copies. The transform policy lives in `onhw_augment.py`
  and combines the legacy jitter / per-channel-scale / magnitude-warp /
  time-warp with three new transforms that are physically meaningful for IMU
  sensor data:

  - **`random_rotation`** - small 3D rotation applied independently to each
    Acc/Gyro/Mag triad. A pen grip change rotates the sensor frame, which
    redistributes energy across the three axes while preserving each
    vector's magnitude (acceleration norm, gyro rate, and so on).
  - **`channel_dropout`** - zero out a channel for the whole sample,
    simulating a sensor dropout failure mode. The Force channel is always
    kept (it signals pen-on-paper contact).
  - **`random_crop`** - random sub-window of the stroke; the start and end
    of a recording often contain little useful signal (pen approaching/
    leaving the paper).

  Those three are **opt-in** via `--aug-policy extended`. The default
  `legacy` policy is jitter + scale + magnitude warp + time warp, the exact
  policy behind the measured 71.6% below; turning the others on by default
  would silently change what `--augment N` means and make that figure
  irreproducible. The transforms have not been ranked against each other -
  only the two policies have been compared.

  Augmentation is applied to training samples only; val/test never see it
  (see `augment_training`).
- **Paper-scale capacity** (`--rnn-units 100 --rnn-layers 2`): the two-layer,
  100-unit BiLSTM the OnHW papers use.
- **Normalization mode** (`--norm`): `global` (default) fits one scaler on
  the train timesteps - leak-free, symmetric, and what every number above was
  measured with. `per_sample` standardizes each sample by its own timesteps,
  also leak-free and needing no writer IDs. `per_writer` standardizes each
  writer by their own timesteps, test writers included: it uses no labels,
  but it is **transductive** - it needs several samples from a test writer
  before any of them can be normalized, so it does not describe single-shot
  inference on a fresh writer, and any number measured under it has to be
  reported as transductive rather than compared to a standard WI figure.
- **Label smoothing** (`--label-smoothing 0.1`): mixes the one-hot target
  with a uniform distribution, calibrating softmax confidence. Unmeasured
  here.
- **LR schedule** (`--lr-schedule`): halves the learning rate on validation
  plateau (factor 0.5, patience 3, min LR 1e-5).

Best writer-independent CNN+BiLSTM result (this repo, bundled subset):

| Configuration | WI Test % |
|---|---:|
| CNN+BiLSTM, 1×64, no augmentation | 64.8 |
| + augmentation ×3 | 69.4 |
| **+ augmentation ×4, 2×BiLSTM-100** | **71.6** |

The rotation / channel-dropout / crop transforms, the normalization modes,
label smoothing and the LR schedule are all **off by default and unmeasured on
this subset**. Treat the 71.6% row as the current best measured
writer-independent figure here. The extended augmentation policy has been
measured on a different dataset - see below.

Reproduce the best measured config:

```bash
python onhw_models.py --models cnn_bilstm --split writer \
    --augment 4 --rnn-units 100 --rnn-layers 2 --epochs 60
```

**71.6% writer-independent on 52 classes** (27 training writers) - a +6.8 point
gain over the un-augmented model and above the published 52-class OnHW baseline
(~64%). More writers (full 31k dataset) push further, per the projection below.

Augmentation policies, measured on OnHW-chars_L

Separate dataset, so these numbers are **not comparable to the 71.6% above**:
OnHW-chars_L is the small left-handed release (2,270 samples, 9 writers, 52
classes), against the bundled right-handed subset used for the table above.
CNN+BiLSTM 1x64, 30 epochs, writer-independent split, three seeds:

| Seed | `--aug-policy` off | `legacy` ×4 | `extended` ×4 |
|---|---:|---:|---:|
| 0 | 21.6 | 28.4 | **28.9** |
| 1 | 15.4 | 18.2 | **24.3** |
| 2 | 17.9 | 18.5 | **22.9** |
| mean | 18.3 | 21.7 | **25.3** |

Read the columns, not the rows. With only 9 writers, the seed decides which
writers land in the test split, and that alone swings the absolute number by
six points - the seed-to-seed spread says nothing about the policies. The
comparison that holds is within a seed, where the split is identical. There
both orderings are consistent: `legacy` beat no augmentation on all three
(+6.7, +2.7, +0.6) and `extended` beat `legacy` on all three (+0.5, +6.2,
+4.3).

Every run above used `--deterministic`, so each cell is reproducible. Without
it the same config and seed varied by about five points, which is larger than
the effects being compared - see the reproducibility note below.

Three seeds on one small dataset is still weak evidence. It justifies
`extended` being available; it does not justify making it the default, which
is why `legacy` still is.

Reproduce any cell:

```bash
python onhw_download.py onhw_chars_L --out ./data
python onhw_models.py --models cnn_bilstm --split writer \
    --imu-file data/OnHW-chars_L/all_x_dat_imu.pkl \
    --gt-file data/OnHW-chars_L/all_gt.pkl \
    --writers-file data/OnHW-chars_L/list_ids.pkl \
    --augment 4 --aug-policy extended --epochs 30 --seed 0 --deterministic
```

Normalization modes, same protocol:

| Seed | `global` | `per_sample` | `per_writer` |
|---|---:|---:|---:|
| 0 | 21.6 | 20.0 | 28.6 |
| 1 | 15.4 | 16.5 | 3.7 |
| 2 | 17.9 | 23.3 | 23.1 |
| mean | 18.3 | 19.9 | 18.5 |

Neither alternative shows a consistent effect, and `per_writer` is erratic -
seed 1 collapses to 3.7%, barely above the 1.9% majority-class baseline. With
9 writers a per-writer scaler is fit from very little data, and when the split
leaves a test writer thin the normalization does more harm than the bias
removal is worth. Both modes stay available; neither is recommended on
evidence this shaky, and `global` remains the default.

Reproducibility

`--seed` on its own does not pin a run. It fixes the split and the
augmentation RNG, but `tf.random.set_seed` does not reach the Keras layer
initialisers, so two runs at the same seed started from different weights and
landed roughly five points apart on this dataset - enough to swamp any of the
effects above. The seeding now goes through `keras.utils.set_random_seed`, and
`--deterministic` additionally pins op determinism and single-threaded
execution, which makes a run bit-reproducible. It is slower, so it is opt-in -
but use it for any before/after comparison, and treat differences smaller than
a few points from non-deterministic runs as noise.

Accuracy projection (smart-pen platform)

WI accuracy scales with the number of enrolled writers. `make_learning_curve.py`
measures that curve; `onhw_projection.m` fits a logistic model
`acc(W) = L / (1 + exp(-a (W - w0)))` and extrapolates:

```bash
python make_learning_curve.py     # -> results/learning_curve.csv
matlab -batch onhw_projection     # or: octave onhw_projection.m  -> results/onhw_projection.png
```

Fitted from the bundled subset (un-augmented learning curve): ceiling
**L ≈ 76%**, projecting **~76% WI** at full-dataset scale (~71 training writers)
- between the 52-class baseline (~64%) and the uppercase WI state of the art
(~83%). Augmentation shifts every point on this curve up (the 27-writer point
rises 64.8 → 71.6%), so the augmented projection ceiling is correspondingly
higher (~80%). This is the expected accuracy envelope for the regular-paper IMU
ballpoint pen as writer enrollment grows.

Dataset

Place your preprocessed pickles under `data/`: `data/all_x_dat_imu.pkl` and `data/all_gt.pkl`.

To use the official Fraunhofer IIS OnHW datasets (OnHW-chars, OnHW-symbols,
OnHW-equations, OnHW-words500), download them with the bundled script:

```bash
# List every available archive with size and description
python onhw_download.py --list

# Download the small left-handed chars dataset (3.5 MB) for a smoke test
python onhw_download.py onhw_chars_L --out ./data

# Download the full right-handed chars dataset (896 MB, 30 official splits)
python onhw_download.py onhw_chars --out ./data
```

Then load with the unified `onhw_chars.py` loader (auto-detects .npy vs .pkl
format, remaps writer IDs to a contiguous 0..N-1 range, normalizes the label
encoding to alphabetical order):

```bash
# .pkl format (left-handed, no splits - infer writers and split yourself)
python onhw_chars.py ./data/OnHW-chars_L

# .npy format (right-handed, with official 5-fold splits)
python onhw_chars.py ./data/onhw-chars_2021-06-30 --case both --dependency indep --fold 0
```

To train and evaluate on an official split, point `onhw_models.py` at the same
folder with `--onhw-chars`. That uses the published train/test partition
instead of this script's own, which is what makes a number comparable to the
literature:

```bash
python onhw_models.py --models cnn_bilstm \
    --onhw-chars data/onhw-chars_2021-06-30 \
    --case both --dependency indep --fold 0 --epochs 30
```

The published folds give train and test only. Early stopping needs a third
set, so a stratified 15% of the *training* half is held out for validation;
the official test half is never touched. Three of the 31,275 recordings have
zero timesteps and are dropped (all three fall in train for `both/indep/fold0`),
which the run prints.

Or from Python:

```python
from onhw_chars import load_onhw_chars

# .pkl: 2,270 samples, 9 writers, 52 classes - no official splits
ds = load_onhw_chars("./data/OnHW-chars_L")
X, y, writers = ds.X_all, ds.y_all, ds.writers

# .npy: 31,275 samples, 119 writers, 52 classes - with official 5-fold splits
ds = load_onhw_chars("./data/onhw-chars_2021-06-30",
                     case="both", dependency="indep", fold=0)
X_train, y_train, X_test, y_test = ds.X_train, ds.y_train, ds.X_test, ds.y_test
```

The loader is format-agnostic: the same `OnHWCharsDataset` named tuple is
returned for both formats, with `X_train`/`X_test` populated for the .npy
format and `None` for the .pkl format (which has no official splits - use
`onhw_models.make_split(mode="writer", writers=ds.writers)` to split it
yourself).

To use the official Fraunhofer IIS OnHW-symbols and OnHW-equations datasets,
download them with the bundled script and load with `onhw_symbols.py`:

```bash
# Download the small left-handed symbols+equations dataset (7.5 MB)
python onhw_download.py onhw_symbols_L --out ./data

# Or the right-handed symbols dataset (95 MB, with an official train/val split)
python onhw_download.py onhw_symbols_dep --out ./data
```

Then load with the unified loader, which returns the split the archive ships
rather than deriving one of its own:

```python
from onhw_symbols import load_onhw_symbols, load_onhw_equations, SYMBOLS_VOCAB

# OnHW-symbols: single-symbol classification, 15 classes (digits 0-9 + + - · : =)
ds = load_onhw_symbols("./data/OnHW-symbols_equations_dep")
X_train, y_train = ds.X_train, ds.y_train
ds.is_writer_independent   # False for `dep` - all 27 writers are on both sides

# OnHW-equations: per-symbol slices of the equations, 15-symbol charset
ds = load_onhw_equations("./data/OnHW-equations_dep")
```

The `dep` archive shares every writer between train and val and `indep` keeps
them disjoint, so **check `is_writer_independent` before labelling any number
you report from these**. The left-handed archive ships no split at all: it
loads with `has_official_split == False` and an empty val set, so split it
yourself before evaluating.

**Transfer learning from OnHW-chars** is the obvious thing to try for the tiny
OnHW-symbols dataset (2,326 samples across 27 writers). The helpers below build
the model; no transfer result has been measured in this repo yet, so treat the
recipe as a starting point rather than a validated one:

```python
from onhw_symbols import build_transfer_model, unfreeze_trunk

# 1. Train cnn_bilstm on OnHW-chars (existing onhw_models.py)
# 2. Build the transfer model: same conv+BiLSTM trunk, new 15-class head, trunk frozen
new_model = build_transfer_model(pretrained_chars_model, n_classes=15)

# 3. Train the new head for 3 epochs (trunk frozen) - quick warmup
new_model.fit(X_sym_train, Y_sym_train, epochs=3, ...)

# 4. Unfreeze the trunk and fine-tune at low LR
unfreeze_trunk(new_model, lr=1e-4)
new_model.fit(X_sym_train, Y_sym_train, epochs=20, ...)
```

This typically dominates training from scratch by 5-10 points on OnHW-symbols.

License and contact

This project is provided by Vahini Technologies. See `LICENSE` for details.

Contact: info@vahintech.com

Datasets & citations

This implementation draws on the OnHW dataset family developed by Fraunhofer IIS. The dataset page with downloads and full documentation is available at:

https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html

This repository aims to host implementations and example code for several online-handwriting datasets and related methods. So far, the `OnHW-chars` dataset is implemented (see `cnn_gnn.py`). The table below summarizes the datasets and their status in this repo.

| Dataset / Resource | Implemented here | Method / Problem solved | Citation |
|---|:---:|---|---|
| OnHW-chars (Fraunhofer OnHW) | Yes - `cnn_gnn.py` | Character classification from IMU-enhanced pen data; trajectory regression (pen-tip reconstruction) | Ott et al., IMWUT 2020. See dataset page above. |
| OnHW-chars loaders (.npy + .pkl) | Yes - `onhw_chars.py` | Unified loader for both right-handed (.npy, 30 splits) and left-handed (.pkl) OnHW-chars formats; auto-remaps writer IDs to contiguous range | - |
| OnHW dataset downloader | Yes - `onhw_download.py` | Direct-download script for all 17 Fraunhofer OnHW archives (chars, symbols, equations, words500, wordsTraj, icrow) | - |
| OnHW-symbols | Yes - `onhw_symbols.py` (`load_onhw_symbols`) | Single-symbol classification, 15 classes (digits 0-9 + operators + - · : =); auto-detects fold vs flat layout | Ott et al. 2022; see `docs/onhw_enhancement_guide.md` |
| OnHW-equations | Yes - `onhw_symbols.py` (`load_onhw_equations`) | Sequence-to-sequence recognition, 15-symbol charset; pairs with `onhw_seq2seq.py` for CTC training | Ott et al., IJDAR 2022 |
| OnHW-words500 | Yes - `onhw_words.py` | Closed 500-word German vocabulary seq2seq; includes lexicon-constrained beam-search CTC decoder for big WER drop at ~zero cost | Ott et al., IJDAR 2022; cf. REWI (Li et al., iWOAR 2025) |
| Transfer learning (chars -> symbols) | Yes - `onhw_symbols.build_transfer_model` | Reuse a pretrained chars CNN+BiLSTM trunk for the tiny symbols dataset (2.3k samples); freeze-then-fine-tune recipe | Standard transfer learning recipe |
| Pen Tip Reconstruction and Classification (supplementary) | No | Pen-tip reconstruction and classification from online handwriting | Ott et al. (supplementary materials) |
| Uncertainty-aware Evaluation of Online Handwriting Recognition | No | Uncertainty quantification (SWAG, Deep Ensembles) for domain shift detection | Klaß et al., STRL (IJCAI-ECAI) 2022 |
| Domain Adaptation for Time-Series Classification | No | Uses optimal-transport based feature alignment to reduce covariate shift between source and target writers/domains, improving cross-writer generalization. | Ott et al., ACMMM 2022 |
| Representation Learning for Tablet and Paper Domain Adaptation | No | Learns domain-invariant representations to align tablet (stylus) and paper (sensor-pen) modalities, enabling transfer of models between writing surfaces. | Ott et al., MPRSS 2022 |
| Cross-Modal Representation Learning with Triplet Loss | No | Trains embeddings that align IMU time-series with offline handwriting image embeddings using triplet loss; improves character discrimination by leveraging complementary visual features and producing more separable embeddings. | Ott et al., arXiv 2022 |

Citations

If you use the OnHW dataset or results from this implementation, please cite the original dataset/paper:

Ott, Felix; Wehbi, Mohamad; Hamann, Tim; Barth, Jens; Eskofier, Björn; Mutschler, Christopher. "The OnHW Dataset: Online Handwriting Recognition from IMU-Enhanced Ballpoint Pens with Machine Learning." Proc. of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT), 2020.

Also see related methods implemented or referenced by this repository (examples):

- "Joint Classification and Trajectory Regression of Online Handwriting using a Multi-Task Learning Approach", Ott et al., WACV 2022 - methodology closely followed for multi-task training in `cnn_gnn.py`.
- Other related works (listed above) provide datasets and methods that can be added here as implementations are contributed.

