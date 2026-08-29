# Benchmark results

Full results, error analysis and the accuracy ceiling for the OnHW-chars
task. The short version lives in the README.

`legacy/cnn_gnn.py` reports ~99% accuracy, but it trains and evaluates on the *same*
array - that figure is train-set memorization, not recognition skill. On a
held-out split the same model scores ~43–47%. `imu2text/models.py` fixes the
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
python -m imu2text.models                 # writer-independent, all 4 models
python -m imu2text.models --split random  # easier random split
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
CNN+BiLSTM's 64.8% WI is close to the 52-class OnHW baseline of the IMWUT 2020
paper (~64%) - on only 27 training writers. The stronger published figure for
the same architecture is 68.06% (Ott et al., ACM MM 2022, Table 3); see the
official-benchmark section below, where the comparison is like for like.

Architectures

![Model architectures](results/architecture.png)

Both stacks are drawn by introspecting the Keras models, so the figure cannot
drift from `imu2text/models.py`. The attention variant differs only after the
BiLSTM: instead of reading out the final state, it keeps every timestep and
learns which ones matter, for 13k extra parameters (145,000 to 157,928).

Transfer learning to OnHW-symbols reuses that trunk:

![Transfer learning](results/transfer_learning.png)

`imu2text.symbols.build_transfer_model` clones the trunk, freezes it, and
attaches a new head, so during the warmup phase 1,515 of 141,263 parameters
train. `unfreeze_trunk()` then releases the rest at a lower learning rate. The
figure reads each layer's `trainable` flag off the model the function actually
returns, so what it labels frozen is frozen. Layout after Figure 6 of Ott et
al., ACM MM 2022 (`data/ACMMM_2022.pdf`); the drawing code is our own.

```bash
python scripts/plot_architecture.py            # both figures
```

Official OnHW-chars benchmark

The bundled subset is a convenience sample. The comparable number comes from
the published dataset and its own splits: 31,275 samples, 119 writers, 52
classes, evaluated on `both/indep/fold0` - the writer-independent protocol the
papers report. Download it once (896 MB) and pass `--onhw-chars`:

```bash
python -m imu2text.download onhw_chars --out ./data
python -m imu2text.models --models cnn_bilstm_attn \
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

**72.5% writer-independent on 52 classes**, 3.3 points above this repo's
previous best configuration, on a model with a third of the parameters of the
2×BiLSTM-100 variants.


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

Single seed, fold 0. The baseline and best rows re-run with `--deterministic`
give 69.29% and 72.26%, both within 0.2 points of the values above.

Machine these numbers were measured on

All figures are CPU-only. TensorFlow reports no GPU visible on this machine.
A CUDA box runs several times faster; the accuracies do not change.

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

Comparing against the published OnHW numbers

The OnHW papers report several tables measuring different tasks.

**Table 3 (ACM MM 2022) is the comparable one.** CRR for right-handed writers
over the same six official splits this repo trains on: {lower, upper,
combined} x {writer-dependent, writer-independent}. Their CNN+BiLSTM row is
the direct competitor.

**Table 4 is not.** Its figures run to 100.00, which is the tell. That table
is a *domain adaptation* benchmark: a model trained on right-handed writers,
carried onto **left-handed** writers using labelled samples from those
writers, and scored on a small left-handed validation set. Its own baseline is
25.19% combined, and the paper's point is that adaptation lifts that to ~96%.
So 85.09 (kMMD, OnHW-symbols) or 100.00 (OnHW-chars lower) answer "how well
does supervised adaptation close a left/right-handed domain gap", not "how
well are characters recognised from an unseen writer". Nothing in this repo
does domain adaptation yet - that is issue #14 - so we have no number that
belongs in Table 4 at all.

### Coverage

The table below is **OnHW-chars only**, and only the right-handed archive.
What each dataset has here today:

| Dataset | Task | Loader | Benchmarked |
|---|---|---|---|
| OnHW-chars (right-handed) | 52/26-class classification | yes | yes, all six splits |
| OnHW-chars_L (left-handed) | same | yes | no |
| OnHW-symbols | 15-class classification | yes | yes, WD and WI |
| OnHW-equations (split) | 15-class, per-symbol slices | yes | yes, WD and WI |
| OnHW-words500 | seq2seq, closed 500-word vocab | yes | no |
| OnHW-wordsRandom | seq2seq, open vocab | no | no |
| OnHW-wordsTraj | seq2seq + trajectory regression | no | no |

Two gaps worth naming.

**Left-handed columns.** Ott et al. report right- and left-handed columns side
by side. The left-handed OnHW-chars archive ships **no official splits**: it is
2,270 samples from 9 writers as flat pickles. Their left-handed WD/WI columns
come from splits they constructed. Any left-handed row here would come from
splits we construct, so it would not be cell-for-cell comparable to theirs
even when the protocol matches. `imu2text/chars.py` loads the archive with real
writer IDs, so `make_split(mode="writer")` can build one.

**Words and trajectories.** `imu2text/words.py` loads OnHW-words500 and
implements lexicon-constrained CTC decoding, but no model has been trained on
it. OnHW-wordsTraj has no loader at all. These are issues
[#11](https://github.com/vahinitech/imu2text/issues/11) and the trajectory
thread in [onhw_research_threads.md](onhw_research_threads.md).

### OnHW-symbols

15 classes (digits 0-9 and + - · : =), 2,326 samples from 27 writers, using
each archive's own train/val split. Published row is Ott et al., ACM MM 2022,
Table 2, right-handed writers.

| Method | WD | WI |
|---|--:|--:|
| CNN+BiLSTM [60] | 96.20 | 79.51 |
| InceptionTime [25] | 91.97 | 76.92 |
| ResNet [86] | 94.50 | 77.41 |
| CNN+BiLSTM+attn (this repo) | 90.49 | 71.52 |
| CNN+BiLSTM+attn, aug x2, LS, LR sched | **95.77** | **72.83** |

Close on writer-dependent (-0.43) and 6.7 points behind on
writer-independent. With 1,575 training samples across 15 classes the WI split
leaves little to generalise from, and the tuned configuration adds only 1.3
points there against 5.3 on WD. The transfer-learning path in
`imu2text/symbols.py` exists for this case and has not been measured.

### OnHW-equations (split)

The `_e` files: 39,643 per-symbol slices cut from 10,713 equations, same
15-symbol charset. Published row is Table 2, right-handed writers.

| Method | WD | WI |
|---|--:|--:|
| CNN+BiLSTM [60] | 95.70 | 83.88 |
| InceptionTime [25] | 94.87 | 84.35 |
| ResNet [86] | 94.68 | 83.45 |
| CNN+BiLSTM+attn (this repo) | 95.33 | **87.04** |
| CNN+BiLSTM+attn, aug x2, LS, LR sched | **96.25** | 86.12 |

The tuned configuration is 0.92 points *behind* the plain one on WI here, the
only split where that happens. With 26,942 training samples this is the largest
classification set in the repo, and augmentation stops paying once the data is
sufficient, matching what the 2xBiLSTM-100 rows show on OnHW-chars.

### Across the three datasets

Writer-independent, ours against the published CNN+BiLSTM row:

| Dataset | Train samples | Published WI | Ours WI | Delta |
|---|--:|--:|--:|--:|
| OnHW-symbols | 1,575 | 79.51 | 72.83 | -6.68 |
| OnHW-chars (52-class) | 19,819 | 68.06 | 72.46 | +4.40 |
| OnHW-equations (split) | 26,942 | 83.88 | 87.04 | +3.16 |

Ahead on the two larger sets, behind on the small one. The changes here are
regularisation, and regularisation needs enough data to regularise: on 1,575
samples across 15 classes it adds 1.3 points where it adds 5.3 on the same
dataset's writer-dependent split. Transfer learning from OnHW-chars is the
path that exists for the small-dataset case and has not been measured.

### OnHW-chars

Six official OnHW-chars splits, fold 0, 30 epochs, single seed. Published rows
are Ott et al., ACM MM 2022, Table 3, right-handed writers.

| Method | Lower WD | Lower WI | Upper WD | Upper WI | Comb WD | Comb WI |
|---|--:|--:|--:|--:|--:|--:|
| CNN+BiLSTM [60] | 88.85 | 79.48 | 92.15 | 85.60 | 78.17 | 68.06 |
| InceptionTime [25] | 84.14 | 75.28 | 87.80 | 81.62 | 70.43 | 61.68 |
| ResNet [86] | 83.01 | 71.93 | 86.41 | 78.03 | 68.56 | 58.74 |
| LSTM-FCN [45] | 81.43 | 71.41 | 85.43 | 77.07 | 67.34 | 57.93 |
| CNN+BiLSTM (this repo) | 85.35 | 80.24 | 88.29 | 82.98 | 75.79 | 67.32 |
| **CNN+BiLSTM+attn, aug x2, LS, LR sched** | 88.25 | **82.45** | 91.43 | **86.78** | **80.07** | **72.46** |

Our plain CNN+BiLSTM reproduces theirs to within 0.7-3.9 points across the six
cells, above on lower WI. The tuned configuration is ahead on all three
writer-independent cells (+2.97, +1.18, +4.40) and on combined WD (+1.90), and
behind by 0.60 and 0.72 on the two writer-dependent single-case cells.

Reproduce our side of Table 3 with:

```bash
python scripts/make_comparison_table.py --config best --epochs 30
```

Where the remaining error actually is

The 52-class figure stopped responding to modelling effort, so the errors were
counted rather than guessed at. `--error-analysis` breaks the test set down by
confusion pair:

```bash
python -m imu2text.models --models cnn_bilstm_attn \
    --onhw-chars data/onhw-chars_2021-06-30 --case both --dependency indep \
    --fold 0 --epochs 30 --augment 2 --aug-policy extended \
    --label-smoothing 0.1 --lr-schedule --error-analysis
```

```
Error analysis on 7956 test samples (2191 wrong, 27.54% error)
case-insensitive accuracy : 84.34%  (plain: 72.46%)
errors that are case only : 945/2191 (43.1% of all errors)
top 12 confusions: 's'->'S' 'o'->'O' 'w'->'W' 'v'->'V' 'z'->'Z' 'u'->'U'
                   'x'->'X' 'c'->'C' 'p'->'P' 'W'->'w' 'Y'->'y' 'y'->'Y'
```

That is the 72.5% model. The weaker 20-epoch baseline had 38.4% of its errors
in case; improving the model from 68.0% to 72.5% pushed the share *up* to
43.1%, because the fixable errors are the ones that got fixed.

![Where the OnHW-chars errors are](results/error_analysis.png)

Rebuild the figure with:

```bash
python -m imu2text.models --models cnn_bilstm_attn \
    --onhw-chars data/onhw-chars_2021-06-30 --case both --dependency indep \
    --fold 0 --epochs 30 --augment 2 --aug-policy extended \
    --label-smoothing 0.1 --lr-schedule \
    --save-predictions results/predictions_official_fold0.npz

python scripts/plot_error_analysis.py \
    --predictions results/predictions_official_fold0.npz \
    --onhw-chars data/onhw-chars_2021-06-30
```

Panel A drops the diagonal so only errors are drawn, and the two red lines mark
where a letter confused with its own other case has to land: 26 off the
diagonal. Almost all of the mass sits on them. Every value is also in
`results/error_analysis_confusions.csv`, so nothing here is readable only by
colour.

Every one of the twelve most common confusions is a letter mistaken for its own
other case. Fold case away and the same model scores 84.3% instead of 72.5%: about twelve
points of the error is nothing but upper-versus-lower.

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
  `imu2text/seq2seq.py` and `docs/onhw_research_threads.md`.
- **Add a sensing modality that sees position** - the tablet-and-camera rig
  behind OnHW-wordsTraj.

Improving accuracy - the available levers

Every configuration below overfits: train accuracy runs 20-30 points above
held-out accuracy on all three datasets below. That gap, not model
capacity, is what limits the numbers. The levers are described here once, and
the sections after measure them, on the official benchmark above and on
OnHW-chars_L further down.

Three datasets appear in this file and their numbers are not
interchangeable. Quote the official benchmark (72.5% WI) against the
literature; the bundled-subset and OnHW-chars_L figures track changes within
this repo.

- **IMU data augmentation** (`--augment N`): each training sample gets `N`
  randomly transformed copies. The transform policy lives in `imu2text/augment.py`
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
python -m imu2text.models --models cnn_bilstm --split writer \
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
python -m imu2text.download onhw_chars_L --out ./data
python -m imu2text.models --models cnn_bilstm --split writer \
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

WI accuracy scales with the number of enrolled writers. `scripts/make_learning_curve.py`
measures that curve; `scripts/onhw_projection.m` fits a logistic model
`acc(W) = L / (1 + exp(-a (W - w0)))` and extrapolates:

```bash
python scripts/make_learning_curve.py     # -> results/learning_curve.csv
matlab -batch scripts/onhw_projection     # or: octave scripts/onhw_projection.m  -> results/onhw_projection.png
```

Fitted from the bundled subset (un-augmented learning curve): ceiling
**L ≈ 76%**, projecting **~76% WI** at full-dataset scale (~71 training writers)
- between the 52-class baseline (~64%) and the uppercase WI state of the art
(~83%). Augmentation shifts every point on this curve up (the 27-writer point
rises 64.8 → 71.6%), so the augmented projection ceiling is correspondingly
higher (~80%). This is the expected accuracy envelope for the regular-paper IMU
ballpoint pen as writer enrollment grows.

