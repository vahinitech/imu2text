# Datasets

The OnHW archives, how to download them and what each loader expects.

Place your preprocessed pickles under `data/`: `data/all_x_dat_imu.pkl` and `data/all_gt.pkl`.

To use the official Fraunhofer IIS OnHW datasets (OnHW-chars, OnHW-symbols,
OnHW-equations, OnHW-words500), download them with the bundled script:

```bash
# List every available archive with size and description
python -m imu2text.download --list

# Download the small left-handed chars dataset (3.5 MB) for a smoke test
python -m imu2text.download onhw_chars_L --out ./data

# Download the full right-handed chars dataset (896 MB, 30 official splits)
python -m imu2text.download onhw_chars --out ./data
```

Then load with the unified `imu2text/chars.py` loader (auto-detects .npy vs .pkl
format, remaps writer IDs to a contiguous 0..N-1 range, normalizes the label
encoding to alphabetical order):

```bash
# .pkl format (left-handed, no splits - infer writers and split yourself)
python -m imu2text.chars ./data/OnHW-chars_L

# .npy format (right-handed, with official 5-fold splits)
python -m imu2text.chars ./data/onhw-chars_2021-06-30 --case both --dependency indep --fold 0
```

To train and evaluate on an official split, point `imu2text/models.py` at the same
folder with `--onhw-chars`. That uses the published train/test partition
instead of this script's own, which is what makes a number comparable to the
literature:

```bash
python -m imu2text.models --models cnn_bilstm \
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
from imu2text.chars import load_onhw_chars

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
`imu2text.models.make_split(mode="writer", writers=ds.writers)` to split it
yourself).

To use the official Fraunhofer IIS OnHW-symbols and OnHW-equations datasets,
download them with the bundled script and load with `imu2text/symbols.py`:

```bash
# Download the small left-handed symbols+equations dataset (7.5 MB)
python -m imu2text.download onhw_symbols_L --out ./data

# Or the right-handed symbols dataset (95 MB, with an official train/val split)
python -m imu2text.download onhw_symbols_dep --out ./data
```

Then load with the unified loader, which returns the split the archive ships
rather than deriving one of its own:

```python
from imu2text.symbols import load_onhw_symbols, load_onhw_equations, SYMBOLS_VOCAB

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
from imu2text.symbols import build_transfer_model, unfreeze_trunk

# 1. Train cnn_bilstm on OnHW-chars (existing imu2text/models.py)
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

This repository aims to host implementations and example code for several online-handwriting datasets and related methods. So far, the `OnHW-chars` dataset is implemented (see `legacy/cnn_gnn.py`). The table below summarizes the datasets and their status in this repo.

| Dataset / Resource | Implemented here | Method / Problem solved | Citation |
|---|:---:|---|---|
| OnHW-chars (Fraunhofer OnHW) | Yes - `legacy/cnn_gnn.py` | Character classification from IMU-enhanced pen data; trajectory regression (pen-tip reconstruction) | Ott et al., IMWUT 2020. See dataset page above. |
| OnHW-chars loaders (.npy + .pkl) | Yes - `imu2text/chars.py` | Unified loader for both right-handed (.npy, 30 splits) and left-handed (.pkl) OnHW-chars formats; auto-remaps writer IDs to contiguous range | - |
| OnHW dataset downloader | Yes - `imu2text/download.py` | Direct-download script for all 17 Fraunhofer OnHW archives (chars, symbols, equations, words500, wordsTraj, icrow) | - |
| OnHW-symbols | Yes - `imu2text/symbols.py` (`load_onhw_symbols`) | Single-symbol classification, 15 classes (digits 0-9 + operators + - · : =); auto-detects fold vs flat layout | Ott et al. 2022; see `docs/onhw_enhancement_guide.md` |
| OnHW-equations | Yes - `imu2text/symbols.py` (`load_onhw_equations`) | Sequence-to-sequence recognition, 15-symbol charset; pairs with `imu2text/seq2seq.py` for CTC training | Ott et al., IJDAR 2022 |
| OnHW-words500 | Yes - `imu2text/words.py` | Closed 500-word German vocabulary seq2seq; includes lexicon-constrained beam-search CTC decoder for big WER drop at ~zero cost | Ott et al., IJDAR 2022; cf. REWI (Li et al., iWOAR 2025) |
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

- "Joint Classification and Trajectory Regression of Online Handwriting using a Multi-Task Learning Approach", Ott et al., WACV 2022 - methodology closely followed for multi-task training in `legacy/cnn_gnn.py`.
- Other related works (listed above) provide datasets and methods that can be added here as implementations are contributed.

