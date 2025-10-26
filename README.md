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

- `cnn_gnn.py` — main script implementing preprocessing, model creation, training and evaluation.
- `LICENSE` — project license and contact information.

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

