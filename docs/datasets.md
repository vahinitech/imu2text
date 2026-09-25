# Datasets

The OnHW archives, how to download them and what each loader expects.

The default run of `imu2text/models.py` reads preprocessed pickles from
`data/`: `data/all_x_dat_imu.pkl` and `data/all_gt.pkl`.

## Downloading

The Fraunhofer IIS OnHW archives (OnHW-chars, OnHW-symbols, OnHW-equations,
OnHW-words500 and others) download with the bundled script:

```bash
# List every available archive with size and description
python -m imu2text.download --list

# Download the small left-handed chars dataset (3.5 MB) for a smoke test
python -m imu2text.download onhw_chars_L --out ./data

# Download the full right-handed chars dataset (896 MB, 30 official splits)
python -m imu2text.download onhw_chars --out ./data
```

## OnHW-chars

`imu2text/chars.py` reads both formats. It detects .npy or .pkl, remaps writer
IDs to a contiguous 0..N-1 range, and re-encodes labels in alphabetical order.

```bash
# .pkl format (left-handed, no splits: infer writers and split yourself)
python -m imu2text.chars ./data/OnHW-chars_L

# .npy format (right-handed, with official 5-fold splits)
python -m imu2text.chars ./data/onhw-chars_2021-06-30 --case both --dependency indep --fold 0
```

The right-handed archive has 30 split directories: 3 cases (`lower`, `upper`,
`both`) × 2 protocols (`dep`, `indep`) × 5 folds.

To train and evaluate on an official split, pass the same folder to
`imu2text/models.py` with `--onhw-chars`. That uses the published train/test
partition instead of a split made here, which is what makes a number
comparable to the literature:

```bash
python -m imu2text.models --models cnn_bilstm \
    --onhw-chars data/onhw-chars_2021-06-30 \
    --case both --dependency indep --fold 0 --epochs 30
```

The published folds give train and test only. Early stopping needs a third
set, so a stratified 15% of the training half is held out for validation; the
official test half is never touched. Three of the 31,275 recordings have zero
timesteps and are dropped (all three fall in train for `both/indep/fold0`),
which the run prints.

From Python:

```python
from imu2text.chars import load_onhw_chars

# .pkl: 2,270 samples, 9 writers, 52 classes, no official splits
ds = load_onhw_chars("./data/OnHW-chars_L")
X, y, writers = ds.X_all, ds.y_all, ds.writers

# .npy: 31,275 samples, 119 writers, 52 classes, with official 5-fold splits
ds = load_onhw_chars("./data/onhw-chars_2021-06-30",
                     case="both", dependency="indep", fold=0)
X_train, y_train, X_test, y_test = ds.X_train, ds.y_train, ds.X_test, ds.y_test
```

Both formats return the same `OnHWCharsDataset` named tuple. `X_train` and
`X_test` are filled for .npy and `None` for .pkl, which has no official split:
use `imu2text.models.make_split(mode="writer", writers=ds.writers)` to make one.

## OnHW-symbols and OnHW-equations

Both load with `imu2text/symbols.py`:

```bash
# Download the small left-handed symbols+equations dataset (7.5 MB)
python -m imu2text.download onhw_symbols_L --out ./data

# Or the right-handed symbols dataset (95 MB, with an official train/val split)
python -m imu2text.download onhw_symbols_dep --out ./data
```

The `dep` and `indep` archives are flat: no fold directories, one official
train/val split each. The loader returns that split as shipped rather than
making its own.

```python
from imu2text.symbols import load_onhw_symbols, load_onhw_equations, SYMBOLS_VOCAB

# OnHW-symbols: single-symbol classification, 15 classes (digits 0-9 + + - · : =)
ds = load_onhw_symbols("./data/OnHW-symbols_equations_dep")
X_train, y_train = ds.X_train, ds.y_train
ds.is_writer_independent   # False for `dep`: all 27 writers are on both sides

# OnHW-equations: per-symbol slices of the equations, 15-symbol charset
ds = load_onhw_equations("./data/OnHW-symbols_equations_dep")
```

`load_onhw_equations` reads the per-symbol `_e` slices, one label per sample.
That is symbol classification, not whole-equation sequence recognition:
putting the equations back together needs the `all_indices_e.txt` mapping,
which the loader does not read yet.

`dep` shares every writer between train and val; `indep` keeps them disjoint.
**Check `is_writer_independent` before labelling any number you report from
these.** The left-handed archive ships no split at all. It loads with
`has_official_split == False` and an empty val set, so split it yourself before
evaluating.

## OnHW-words500

`imu2text/words.py` loads the Words500 archives and implements
lexicon-constrained CTC beam search (`LexiconDecoder`). The vocabulary is 500
closed German words over a 59-character charset: A-Z, a-z and ÄÖÜäöüß.

## Transfer learning from OnHW-chars

OnHW-symbols is small (2,326 samples across 27 writers), so reusing a trunk
trained on OnHW-chars is the obvious thing to try. The helpers below build the
model. No transfer result has been measured here yet, so this is a starting
point, not a validated recipe:

```python
from imu2text.symbols import build_transfer_model, unfreeze_trunk

# 1. Train cnn_bilstm on OnHW-chars (existing imu2text/models.py)
# 2. Build the transfer model: same conv+BiLSTM trunk, new 15-class head, trunk frozen
new_model = build_transfer_model(pretrained_chars_model, n_classes=15)

# 3. Train the new head for 3 epochs (trunk frozen) as a quick warmup
new_model.fit(X_sym_train, Y_sym_train, epochs=3, ...)

# 4. Unfreeze the trunk and fine-tune at low LR
unfreeze_trunk(new_model, lr=1e-4)
new_model.fit(X_sym_train, Y_sym_train, epochs=20, ...)
```

Run both this and training from scratch before quoting a gain.

## What is implemented

The OnHW dataset family comes from Fraunhofer IIS. Downloads and full
documentation are on the dataset page:
https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html

| Dataset / resource | Here | What it does | Citation |
|---|:---:|---|---|
| OnHW-chars | Yes, `legacy/cnn_gnn.py` | Character classification plus pen-tip trajectory regression (multi-task) | Ott et al., IMWUT 2020 |
| OnHW-chars loaders (.npy + .pkl) | Yes, `imu2text/chars.py` | Right-handed .npy (30 split directories) and left-handed .pkl; remaps writer IDs to a contiguous range | - |
| OnHW downloader | Yes, `imu2text/download.py` | Downloads all 17 Fraunhofer OnHW archives (chars, symbols, equations, words500, wordsTraj, icrow) | - |
| OnHW-symbols | Yes, `imu2text/symbols.py` (`load_onhw_symbols`) | Single-symbol classification, 15 classes (digits 0-9 and + - · : =); flat dep/indep archives with one official split, or the unsplit left-handed archive | Ott et al. 2022; see [roadmap.md](roadmap.md) |
| OnHW-equations | Yes, `imu2text/symbols.py` (`load_onhw_equations`) | Symbol classification on the per-symbol `_e` slices, 15 classes. Whole-equation recognition needs `all_indices_e.txt`, which the loader does not read yet | Ott et al., IJDAR 2022 |
| OnHW-words500 | Yes, `imu2text/words.py` | Seq2seq over a closed 500-word German vocabulary, 59-character charset; lexicon-constrained beam-search CTC decoder | Ott et al., IJDAR 2022; cf. REWI (Li et al., iWOAR 2025) |
| Transfer learning (chars to symbols) | Yes, `imu2text.symbols.build_transfer_model` | Reuses a pretrained chars CNN+BiLSTM trunk for the symbols dataset (2.3k samples); freeze, then fine-tune | Standard transfer-learning recipe |
| Pen tip reconstruction and classification (supplementary) | No | Pen-tip reconstruction and classification from online handwriting | Ott et al. (supplementary materials) |
| Uncertainty-aware evaluation of online handwriting recognition | No | Uncertainty estimates (SWAG, deep ensembles) for detecting domain shift | Klaß et al., STRL (IJCAI-ECAI) 2022 |
| Domain adaptation for time-series classification | No | Optimal-transport feature alignment to reduce covariate shift between source and target writers | Ott et al., ACMMM 2022 |
| Representation learning for tablet and paper domain adaptation | No | Domain-invariant representations shared by tablet (stylus) and paper (sensor pen) writing | Ott et al., MPRSS 2022 |
| Cross-modal representation learning with triplet loss | No | Triplet loss aligning IMU time-series embeddings with offline handwriting image embeddings | Ott et al., arXiv 2022 |

## Citations

If you use the OnHW dataset or results from this repo, cite the original paper:

Ott, Felix; Wehbi, Mohamad; Hamann, Tim; Barth, Jens; Eskofier, Björn; Mutschler, Christopher. "The OnHW Dataset: Online Handwriting Recognition from IMU-Enhanced Ballpoint Pens with Machine Learning." Proc. of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT), 2020.

The multi-task training in `legacy/cnn_gnn.py` follows "Joint Classification
and Trajectory Regression of Online Handwriting using a Multi-Task Learning
Approach", Ott et al., WACV 2022.
