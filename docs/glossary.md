# Glossary

Abbreviations used in this codebase and in the OnHW papers it builds on.
Terms that appear as a CLI flag or a module here are marked with the place to
look.

## The project

| Term | Meaning |
|---|---|
| **OnHW** | Online HandWriting. "Online" means the signal is captured while writing (a time series), as opposed to *offline* recognition from an image of finished text. The Fraunhofer IIS dataset family: OnHW-chars, -symbols, -equations, -words500, -wordsRandom, -wordsTraj. |
| **IMU** | Inertial Measurement Unit. The accelerometers and gyroscope in the pen. Here each sample is 13 channels at 100 Hz: two 3-axis accelerometers, a 3-axis gyroscope, a 3-axis magnetometer, and one force channel. |
| **MTS** | Multivariate Time Series. Several signals sampled over the same time axis. One pen recording is an MTS of shape (timesteps, 13). |
| **MTSC** | MTS Classification. Assigning one label to a whole time series, e.g. which character was written. |
| **HAR** | Human Activity Recognition. The neighbouring field the OnHW methods borrow from. |

## Datasets and evaluation

| Term | Meaning |
|---|---|
| **WD** | Writer-Dependent. The same writers appear in train and test. Easier: the model can learn a person's style. |
| **WI** | Writer-Independent. Whole writers are held out, so every test writer is unseen. The protocol the OnHW papers report and the default here (`--dependency indep`, or `--split writer`). |
| **Fold** | One of the pre-computed train/test partitions shipped with a dataset. OnHW-chars has 30: {lower, upper, both} x {dep, indep} x 5 folds. |
| **Split** | A partition of the data into train / validation / test. "The published split" is the one in the archive; "constructed" means this repo made its own. Always state which. |
| **Combined / both** | The 52-class task, A-Z and a-z together. `lower` and `upper` are 26 classes each. |
| **Left/right-handed** | OnHW ships separate archives. Right-handed sets are large; left-handed are small, which is why they are used as the source domain in the domain-adaptation work. |
| **CRR** | Character Recognition Rate, in %. The OnHW papers' classification metric. Same thing as accuracy for single-character tasks. |
| **CER / WER** | Character / Word Error Rate. Edit distance divided by reference length, for sequence tasks. Lower is better. See `imu2text/seq2seq.py`. |

## Architectures

| Term | Meaning |
|---|---|
| **CNN** | Convolutional Neural Network. Slides learned filters over the signal to pick up local shapes. Here it is the trunk that reads short stroke patterns and downsamples time. |
| **LSTM** | Long Short-Term Memory. A recurrent layer that carries state along the sequence, so later timesteps can depend on earlier ones. |
| **BiLSTM** | Bidirectional LSTM. Two LSTMs, one reading forwards and one backwards, concatenated. A stroke's meaning often depends on what comes after it, which a forward-only pass cannot see. |
| **CNN+BiLSTM** | The OnHW baseline and this repo's main model: CNN trunk for local features, BiLSTM for temporal context. `--models cnn_bilstm`. |
| **Attention pooling** | Instead of reading only the BiLSTM's final state, keep every timestep and learn a weight for each, then take the weighted average. `--models cnn_bilstm_attn`. |
| **GNN / GCN** | Graph Neural Network / Graph Convolutional Network. Used in `legacy/cnn_gnn.py`, which treats sensor channels as graph nodes. Reference only. |
| **FCN** | Fully Convolutional Network. No dense layers; LSTM-FCN combines one with an LSTM branch. |
| **ResNet** | Residual Network. Adds skip connections so gradients reach early layers in a deep stack. XResNet and ResCNN are variants. |
| **InceptionTime / XceptionTime** | Time-series adaptations of the Inception and Xception image architectures: parallel convolutions at several kernel sizes. |
| **TST** | Time Series Transformer. Self-attention instead of recurrence. |
| **BN / BNorm** | Batch Normalization. Rescales activations per mini-batch, which stabilises and speeds up training. |
| **Dropout** | Randomly zeroes activations during training to discourage memorisation. Inactive at inference. |

## Training

| Term | Meaning |
|---|---|
| **CTC** | Connectionist Temporal Classification. Trains a sequence model when you know the target string but not which timestep produced which symbol. It sums over every alignment and uses a **blank** symbol to separate repeats. See `imu2text/seq2seq.py` and `imu2text/words.py`. |
| **Blank** | The extra CTC symbol meaning "emit nothing here". Also used as padding in the shipped Words500 labels, which is why the loader strips it. |
| **Greedy decoding** | Take the most likely symbol at each frame, collapse repeats, drop blanks. Fast, no lookahead. |
| **Beam search** | Keep the best *k* partial hypotheses at each step instead of one. `LexiconDecoder` also prunes to prefixes of a known vocabulary. |
| **Lexicon** | A closed set of allowed outputs, e.g. the 500 words of OnHW-words500. Constraining the decode to it removes every non-word. |
| **Label smoothing** | Replaces the one-hot target with a slightly softened distribution, so the model is not pushed to be maximally confident. `--label-smoothing 0.1`. |
| **LR** | Learning Rate. `--lr-schedule` halves it when validation accuracy stops improving. |
| **Early stopping** | Stop when validation stops improving and restore the best weights, rather than training a fixed number of epochs. |
| **Augmentation** | Synthetic training copies made by transforming real ones: jitter, scaling, warping, rotation. Train split only. See `imu2text/augment.py`. |
| **Epoch** | One pass over the training set. |
| **MTL / STL** | Multi-Task / Single-Task Learning. MTL trains one model on several objectives at once, e.g. classify the character *and* reconstruct the pen trajectory. |
| **DWA** | Dynamic Weight Averaging. Balances MTL task weights by how fast each loss is falling. |

## Losses

| Term | Meaning |
|---|---|
| **Cross-entropy** | The standard classification loss: penalises probability assigned to the wrong class. |
| **MSE** | Mean Squared Error. Average squared distance between prediction and target. |
| **Huber** | Squared error near zero, absolute error further out, so single outliers matter less. |
| **Distance loss** | Any loss aligning predicted and true positions pointwise (MSE, Huber, Andrew's Sine). Aligns location but can dilate the shape. |
| **Similarity loss** | Cosine similarity or Pearson correlation. Matches the *shape* of two sequences, invariant to scale (and, for Pearson, to shift). Aligns shape but not size. |
| **Wasserstein** | Distance between two probability distributions, defined as the cheapest way to transport one into the other. Also called earth mover's distance. |
| **Soft-DTW** | A differentiable relaxation of DTW, usable as a training loss. |

Distance and similarity losses pull in different directions, which is the
problem Ott et al. (WACV 2022, `data/WACV2022_paper.pdf`) address with
multi-task weighting.

## Domain adaptation

Terms from Ott et al., ACM MM 2022 (`data/ACMMM_2022.pdf`). Nothing in this
repo implements these yet.

| Term | Meaning |
|---|---|
| **DA** | Domain Adaptation. Making a model trained on one distribution work on a related but different one, e.g. right-handed writers to left-handed. |
| **TL** | Transfer Learning. Reusing a trained model on a new task, usually by freezing most layers and retraining the head. See `imu2text/symbols.build_transfer_model`. |
| **Source / target domain** | In that paper the target is the large right-handed data the model was trained on, and the source is the small left-handed data being adapted to. Note the naming is the reverse of some DA literature. |
| **Covariate shift** | The inputs change distribution between domains while the labels mean the same thing. |
| **OT** | Optimal Transport. The cheapest way to move one distribution onto another. |
| **EMD** | Earth Mover's Distance. The classical OT distance. |
| **SEMD** | The Laplacian-regularised OT variant used beside EMD in that paper's tables (their reference [29], Flamary et al., 2014). |
| **Sinkhorn** | OT with an entropy penalty, which makes it much faster to solve. |
| **MMD** | Maximum Mean Discrepancy. Distance between two distributions measured through their mean embeddings in a kernel space. |
| **kMMD** | Kernel MMD. Equivalent to first-order kernelised HoMM. |
| **HoMM** | Higher-order Moment Matching. Matches moments beyond the mean. Order 1 is equivalent to MMD, order 2 to CORAL. |
| **CORAL** | CORrelation ALignment. Aligns the covariance of source and target features. Jeffrey (J) and Stein (S) are variants using different divergences. |
| **CC / PC** | Cross Correlation / Pearson Correlation, used as similarity metrics for selecting a transformation. |
| **RKHS** | Reproducing Kernel Hilbert Space. The feature space a kernel implicitly maps into. |

## Metrics and statistics

| Term | Meaning |
|---|---|
| **AUC** | Area Under the ROC Curve. Read here as: the probability a random positive scores above a random negative. 0.5 means the feature cannot separate the two groups. |
| **ECE** | Expected Calibration Error. How far a model's confidence is from its accuracy. A well-calibrated model that says 70% is right 70% of the time. |
| **MC-dropout** | Monte Carlo dropout. Keep dropout on at inference and predict several times; the spread estimates uncertainty. |
| **DTW** | Dynamic Time Warping. Distance between two sequences that allows stretching along the time axis, so the same shape written at different speeds still matches. |
| **kNN** | k-Nearest Neighbours. Classify by the labels of the closest training examples. DTW-kNN is the classical handwriting baseline. |
| **t-SNE** | A method for projecting high-dimensional features to 2D for inspection. Distances in the plot are not quantitative. |
| **CVD** | Colour Vision Deficiency. The figure scripts check palettes against it. |

## Repo conventions

| Term | Where |
|---|---|
| `--deterministic` | Makes a run bit-reproducible: op determinism plus single-threaded execution. Needed for any before/after comparison. |
| `--norm` | `global` (default), `per_sample`, or `per_writer`. The last is transductive and any result from it must say so. |
| `--aug-policy` | `legacy` (jitter, scale, magnitude warp, time warp) or `extended` (adds rotation, channel dropout, crop). |
| **Transductive** | The method needs several samples from the test subject before it can process any of them. Not comparable to single-shot inference on a new writer. |
| **Leakage** | Any path by which information about the test set reaches training, e.g. fitting normalisation on all data instead of the train split. |
