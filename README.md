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

`requirements.txt` pins patched releases for all GitHub security advisories
reported against this repo: `torch==2.7.1` (fixes the critical
`torch.load`-with-`weights_only=True` RCE, CVE-2025-32434, plus the 2025
memory-corruption/DoS advisories) and `scikit-learn==1.5.2` (CVE-2024-5206).
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

Improving accuracy - augmentation + capacity

The small writer-independent training set overfits (train ≈ 96%, WI ≈ 65%). Three
honest levers close part of that gap:

- **IMU data augmentation** (`--augment N`): each training sample gets `N`
  randomly transformed copies. The transform policy lives in `onhw_augment.py`
  and combines the legacy jitter / per-channel-scale / magnitude-warp /
  time-warp with three new transforms that are physically meaningful for IMU
  sensor data:

  - **`random_rotation`** - small 3D rotation applied independently to each
    Acc/Gyro/Mag triad. A pen grip change rotates the sensor frame, which
    redistributes the energy across the three axes while preserving the
    vector magnitude (acceleration norm, gyro rate, etc.). This is the
    single most impactful IMU-specific transform.
  - **`channel_dropout`** - zero out a channel for the whole sample,
    simulating a sensor dropout failure mode. The Force channel is always
    kept (it signals pen-on-paper contact).
  - **`random_crop`** - random sub-window of the stroke; the start and end
    of a recording often contain little useful signal (pen approaching/
    leaving the paper).

  Augmentation is applied to training samples only; val/test never see it
  (see `augment_training`).
- **Paper-scale capacity** (`--rnn-units 100 --rnn-layers 2`): the two-layer,
  100-unit BiLSTM the OnHW papers use.
- **Per-writer normalization** (`--per-writer-norm`): a separate scaler is
  fit for each training writer from their own timesteps, removing per-writer
  sensor-mount/grip bias. Unseen test writers fall back to the global train
  scaler - no test data leaks into training.
- **Label smoothing** (`--label-smoothing 0.1`): mixes the one-hot target
  with a uniform distribution, calibrating softmax confidence and typically
  adding ~0.5-1.0 points on the 52-class task.
- **LR schedule** (`--lr-schedule`): halves the learning rate on validation
  plateau (factor 0.5, patience 3, min LR 1e-5).

Best writer-independent CNN+BiLSTM result (this repo, bundled subset):

| Configuration | WI Test % |
|---|---:|
| CNN+BiLSTM, 1×64, no augmentation | 64.8 |
| + augmentation ×3 | 69.4 |
| + augmentation ×4, 2×BiLSTM-100 | 71.6 |
| + per-writer norm + label smoothing 0.1 + LR schedule | ~73 (expected) |

Reproduce the best config:

```bash
python onhw_models.py --models cnn_bilstm --split writer \
    --augment 4 --rnn-units 100 --rnn-layers 2 --epochs 60 \
    --per-writer-norm --label-smoothing 0.1 --lr-schedule
```

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
python onhw_chars.py ./data/OnHW-chars_2021-06-30 --case both --dependency indep --fold 0
```

Or from Python:

```python
from onhw_chars import load_onhw_chars

# .pkl: 2,270 samples, 9 writers, 52 classes - no official splits
ds = load_onhw_chars("./data/OnHW-chars_L")
X, y, writers = ds.X_all, ds.y_all, ds.writers

# .npy: 31,275 samples, 119 writers, 52 classes - with official 5-fold splits
ds = load_onhw_chars("./data/OnHW-chars_2021-06-30",
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

# Or the full right-handed symbols dataset (95 MB, with 5-fold splits)
python onhw_download.py onhw_symbols_dep --out ./data
```

Then load with the unified loader (auto-detects fold-based vs flat layout,
handles both symbols and equations sub-datasets):

```python
from onhw_symbols import load_onhw_symbols, load_onhw_equations, SYMBOLS_VOCAB

# OnHW-symbols: single-symbol classification, 15 classes (digits 0-9 + + - · : =)
ds = load_onhw_symbols("./data/OnHW-symbols_equations_dep", fold=0)
X_train, y_train = ds.X_train, ds.y_train

# OnHW-equations: sequence-to-sequence, 15-symbol charset
ds = load_onhw_equations("./data/OnHW-equations_dep", fold=0)
# Use onhw_seq2seq for CTC training on these
```

**Transfer learning from OnHW-chars** is the recommended recipe for the tiny
OnHW-symbols dataset (2,326 samples is too small to train a CNN+BiLSTM from
scratch). Pretrain on OnHW-chars (31k samples, 119 writers), then transfer:

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

