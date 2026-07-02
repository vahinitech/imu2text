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

- `cnn_gnn.py` — original single-script example (preprocessing, model, train/eval).
- `onhw_models.py` — honest OnHW benchmark suite: baselines (CNN, LSTM, BiLSTM) and the SOTA CNN+BiLSTM, with both writer-independent and random splits.
- `make_learning_curve.py` — trains the SOTA model on an increasing number of writers (writer-independent) to produce the accuracy learning curve.
- `onhw_projection.m` — MATLAB/Octave script that fits a logistic model to the learning curve and projects pen accuracy to full-dataset scale.
- `LICENSE` — project license and contact information.

Benchmarks (honest, held-out evaluation)

`cnn_gnn.py` reports ~99% accuracy, but it trains and evaluates on the *same*
array — that figure is train-set memorization, not recognition skill. On a
held-out split the same model scores ~43–47%. `onhw_models.py` fixes the
methodology (real train/val/test split, train-only normalization, early
stopping) and implements the OnHW baselines plus the SOTA CNN+BiLSTM.

Two evaluation protocols are supported:

- `--split writer` (default) — **writer-independent (WI)**: whole writers are
  held out, so test writers are entirely unseen. This is the protocol the OnHW
  papers report. Writer IDs are reconstructed from the recording order (the pen
  records one writer's full alphabet at a time — see `infer_writer_ids`).
- `--split random` — stratified random (easier; a writer's style can leak into
  both train and test).

```bash
python onhw_models.py                 # writer-independent, all 4 models
python onhw_models.py --split random  # easier random split
```

Writer-independent results on the bundled subset (2,270 samples, 45 writers,
52 classes; 1×BiLSTM-64, seq len 100 — CPU-friendly config):

| Model       | Train % | WI Test % |
|-------------|--------:|----------:|
| **cnn_bilstm** (SOTA) | 96.1 | **64.8** |
| bilstm      | 88.7 | 56.2 |
| cnn         | 90.8 | 48.7 |
| lstm        | 75.7 | 43.8 |
| majority baseline | — | 2.2 |

The ordering matches the literature (CNN+BiLSTM > BiLSTM > CNN > LSTM), and
CNN+BiLSTM's 64.8% WI essentially matches the published 52-class OnHW baseline
(~64%, Ott et al. 2020) — on only 27 training writers.

Improving accuracy — augmentation + capacity

The small writer-independent training set overfits (train ≈ 96%, WI ≈ 65%). Two
honest levers close part of that gap:

- **IMU data augmentation** (`--augment N`): each training sample gets `N`
  randomly transformed copies — per-channel scaling, std-proportional jitter,
  smooth magnitude warp, and time warp. Augmentation is applied to training
  samples only; val/test never see it (see `augment_training`).
- **Paper-scale capacity** (`--rnn-units 100 --rnn-layers 2`): the two-layer,
  100-unit BiLSTM the OnHW papers use.

Best writer-independent CNN+BiLSTM result (this repo, bundled subset):

| Configuration | WI Test % |
|---|---:|
| CNN+BiLSTM, 1×64, no augmentation | 64.8 |
| + augmentation ×3 | 69.4 |
| **+ augmentation ×4, 2×BiLSTM-100** | **71.6** |

Reproduce the best config:

```bash
python onhw_models.py --models cnn_bilstm --split writer \
    --augment 4 --rnn-units 100 --rnn-layers 2 --epochs 60
```

**71.6% writer-independent on 52 classes** (27 training writers) — a +6.8 point
gain over the un-augmented model and above the published 52-class OnHW baseline
(~64%). More writers (full 31k dataset) push further, per the projection below.

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
— between the 52-class baseline (~64%) and the uppercase WI state of the art
(~83%). Augmentation shifts every point on this curve up (the 27-writer point
rises 64.8 → 71.6%), so the augmented projection ceiling is correspondingly
higher (~80%). This is the expected accuracy envelope for the regular-paper IMU
ballpoint pen as writer enrollment grows.

Dataset

Place your preprocessed pickles under `data/`: `data/all_x_dat_imu.pkl` and `data/all_gt.pkl`.

License and contact

This project is provided by Vahini Technologies. See `LICENSE` for details.

Contact: info@vahintech.com

Datasets & citations

This implementation draws on the OnHW dataset family developed by Fraunhofer IIS. The dataset page with downloads and full documentation is available at:

https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html

This repository aims to host implementations and example code for several online-handwriting datasets and related methods. So far, the `OnHW-chars` dataset is implemented (see `cnn_gnn.py`). The table below summarizes the datasets and their status in this repo.

| Dataset / Resource | Implemented here | Method / Problem solved | Citation |
|---|:---:|---|---|
| OnHW-chars (Fraunhofer OnHW) | Yes — implemented in `cnn_gnn.py` | Character classification from IMU-enhanced pen data; trajectory regression (pen-tip reconstruction) | Ott et al., IMWUT 2020. See dataset page above. |
| Pen Tip Reconstruction and Classification (supplementary) | No | Pen-tip reconstruction and classification from online handwriting | Ott et al. (supplementary materials) |
| Uncertainty-aware Evaluation of Online Handwriting Recognition | No | Uncertainty quantification (SWAG, Deep Ensembles) for domain shift detection | Klaß et al., STRL (IJCAI-ECAI) 2022 |
| Domain Adaptation for Time-Series Classification | No | Uses optimal-transport based feature alignment to reduce covariate shift between source and target writers/domains, improving cross-writer generalization. | Ott et al., ACMMM 2022 |
| Representation Learning for Tablet and Paper Domain Adaptation | No | Learns domain-invariant representations to align tablet (stylus) and paper (sensor-pen) modalities, enabling transfer of models between writing surfaces. | Ott et al., MPRSS 2022 |
| Cross-Modal Representation Learning with Triplet Loss | No | Trains embeddings that align IMU time-series with offline handwriting image embeddings using triplet loss; improves character discrimination by leveraging complementary visual features and producing more separable embeddings. | Ott et al., arXiv 2022 |

| Sequence-based OnHW Datasets | No | Sequence datasets (words, equations, multi-character streams) for sequence-to-sequence and CTC-style models. These require sequence models (seq2seq, CTC, or Transformer) and may include writer-dependent / writer-independent splits. | Ott et al., IJDAR 2022 |

Citations

If you use the OnHW dataset or results from this implementation, please cite the original dataset/paper:

Ott, Felix; Wehbi, Mohamad; Hamann, Tim; Barth, Jens; Eskofier, Björn; Mutschler, Christopher. "The OnHW Dataset: Online Handwriting Recognition from IMU-Enhanced Ballpoint Pens with Machine Learning." Proc. of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT), 2020.

Also see related methods implemented or referenced by this repository (examples):

- "Joint Classification and Trajectory Regression of Online Handwriting using a Multi-Task Learning Approach", Ott et al., WACV 2022 — methodology closely followed for multi-task training in `cnn_gnn.py`.
- Other related works (listed above) provide datasets and methods that can be added here as implementations are contributed.

