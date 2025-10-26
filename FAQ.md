# FAQ — imu2text (beginner-friendly)

This document collects common questions and short, practical answers for new users of the imu2text repository. It focuses on how the example model (`cnn_gnn.py`) works, how it relates to sequence vs. character datasets, which models are suitable for sensor pens, how to load data, and recommended hardware.

---

Table of contents

- Introduction / quick start
- Sensor basics (accelerometer, gyroscope, magnetometer, force)
- Character vs sequence datasets
- Offline vs online (sensor and image) workflows
- What to learn (algorithms & skills)
- How sensor data helps recognition
- Advantages / disadvantages (IMU pens, OCR, online vs offline)
- Datasets, loading, splits
- Hardware recommendations
- Quick rules-of-thumb


Q: If I write with a sensor pen, will this model detect all characters I write?

A: Short answer: not automatically. The model in `cnn_gnn.py` is a character-level, multi-task example: it expects one recorded IMU sample per character (an isolated character stroke sequence) and predicts one character label per sample. It will only recognize characters that were present in the training data and that the model was trained to output (for example, A–Z). If you write continuous text (multiple characters) without segmentation, you need either:

- a segmentation step that splits the continuous IMU stream into character-level samples (then feed each to the character model), or
- a sequence model (sequence-to-sequence, CTC, or Transformer) trained to take a long IMU stream and produce a sequence of characters directly.

Robust recognition also depends on training data diversity (writers, styles), data preprocessing, and suitable modeling choices.

---

Q: What's the difference between "sequence" and "character" OnHW datasets?

A:
- Character datasets (e.g. `OnHW-chars`) contain isolated characters. Each example corresponds to one character stroke and is suitable for per-character classifiers (CNN, CNN+GNN, small RNNs). Training is simpler and evaluation is per-character accuracy.
- Sequence datasets (OnHW sequence splits: words, equations, multi-character streams) contain continuous recordings of multiple characters. These require sequence models and special training objectives (CTC, seq2seq with attention, or Transformer-based encoders/decoders). Sequence datasets often provide writer-dependent and writer-independent splits and are used to evaluate end-to-end handwriting recognition.

In short: use character models for isolated glyph classification and sequence models for continuous handwriting recognition.

---


Q: Can I use this model for offline character recognition (images) or for offline processing of sensor feeds?

A: Two different meanings of "offline" arise; both are possible but distinct:

- Offline images (scanned handwriting): This is an image-based OCR problem. `cnn_gnn.py` is not built for images; use image-based CNN/CRNN/OCR models for scanned handwriting. Cross-modal methods (triplet/contrastive losses) can later help align image and IMU embeddings if you have paired data.

- Offline sensor-fed recognition (record then process): Yes — you can record IMU data from the pen, save it, and run recognition later (batch/offline inference). In this case you still use time-series models (CNN/RNN/Transformer/GCN) but run them as post-processing instead of streaming in real time. Offline processing has advantages: you can run heavier models, perform more robust segmentation, and apply offline post-processing (language models, beam search) to improve accuracy.


Practical notes for sensor-fed offline workflows:

- Preserve timestamps and ordering when recording.
- Use segmentation or an end-to-end sequence model for continuous recordings.
- Offline processing allows heavier preprocessing (filtering, sensor fusion) and larger models (Transformers, ensembles).

---

Q: Basic sensor overview — what are accelerometers, gyroscopes, magnetometers and force sensors?

A: Short beginner-friendly definitions:

- Accelerometer: measures linear acceleration along 3 axes (gives speed and direction changes). Useful to infer movement and sudden changes.
- Gyroscope: measures angular velocity (rotation) around three axes. Useful to detect pen rotations, spin and orientation changes.
- Magnetometer: measures the magnetic field (used to estimate heading / orientation relative to Earth’s magnetic field).
- Force / pressure sensor: measures pressure applied by the pen tip — useful to detect pen contact, pen lifts, and stroke emphasis.

Together these sensors provide a multi-channel time-series describing how the pen moved and contacted the surface.

---

Q: Advantage and disadvantage of offline character recognition with sensor pen

A: Offline character recognition with sensor pen means you record sensor data and process it later (not necessarily images). Advantages:

- You can run heavier models and more complex preprocessing offline.
- Easier to batch-process, label, and re-run experiments (good for research).
- You can apply stronger post-processing (language models, beam search) and multi-pass segmentation.

Disadvantages:

- Requires collecting and storing sensor data — more complex data management.
- No immediate feedback to the user (not real-time).
- Errors from segmentation or drift may only be noticed after recording; real-time corrections are not possible.


---

Q: What algorithms should a new learner study to solve sensor-pen handwriting recognition problems?

A: Key topics and algorithms to learn (ordered roughly from foundational to advanced):

- Signal processing basics: filtering (low/high-pass), resampling, interpolation, denoising, and simple feature extraction (velocity, acceleration magnitude, tilt angles).
- Time-series basics: normalization, windowing, sequence padding/truncation, and sequence augmentation (time warping, jittering).
- Classical ML: k-NN, SVMs for baseline experiments on hand-crafted features.
- Deep learning for time series:
  - 1D Convolutional Neural Networks (CNNs) for local feature extraction.
  - Recurrent Neural Networks (LSTM/GRU) for temporal modeling.
  - Transformer encoders/decoders for sequence-to-sequence tasks.
  - Connectionist Temporal Classification (CTC) and seq2seq + attention for unsegmented sequence transcription.
- Graph Neural Networks (GNNs): when you want to model relationships between time steps or sensor channels as a graph.
- Representation learning / metric learning: triplet loss, contrastive learning — useful for cross-modal alignment (IMU ↔ images) or few-shot learning.
- Domain adaptation & robustness: optimal transport, adversarial adaptation, SWAG/deep ensembles for uncertainty estimation.
- Practical skills: data pipelines, batching/padding, evaluation metrics (accuracy, CER/WER for sequences), and building basic GUIs or recording tools to collect labeled data.

Resources: follow short tutorials on each (Keras/TensorFlow or PyTorch), try small projects (classify a few characters), and progressively increase model complexity.

---

Q: How does sensor data help character recognition? What information does it provide?

A: IMU and pen sensors provide dynamic, complementary information that images alone do not:

- Temporal dynamics: the order, timing, and speed of strokes — useful to disambiguate visually similar characters.
- 3D motion cues: accelerometer and gyroscope give movement and orientation, which help recover pen strokes even if the pen slips or pressure varies.
- Pressure/force: pen pressure indicates pen contact and can help segment strokes and detect pen lifts.
- Tilt and orientation: help infer stroke direction and pen pose, improving stroke modeling.
- Robustness to occlusion: sensor data is unaffected by paper stains, low contrast, or poor lighting that can hurt image OCR.

Together, these channels let models learn the generative process of handwriting (how strokes are produced), not just the final ink pattern — which often improves recognition, especially for ambiguous glyphs or stylized writing.

---

Q: What are the advantages and disadvantages of IMU sensor pens?

A: Advantages
- Rich temporal and motion data (acceleration, rotation, pressure).
- Works in low-light or on surfaces where image capture is hard.
- Enables real-time feedback and interactive applications.
- Can separate pen motion from background noise (no need for perfect scans).

Disadvantages
- Adds hardware cost and requires calibrated sensors.
- Sensor drift, noise, and channel misalignment require preprocessing and calibration.
- Data formats and sampling rates vary by device; collecting labeled data is harder than curating images.
- Not all handwriting uses distinct motion patterns (some differences are subtle), and inter-writer variability is high.

---

Q: Advantages and disadvantages — online (sensor/real-time) vs offline (image) handwriting recognition

A: Online (sensor-based) advantages:
- Captures stroke dynamics, pressure, and timing — improves disambiguation.
- Enables immediate feedback and interactive apps (handwriting-to-text in real-time).
- Less sensitive to paper/scan quality.

Online disadvantages:
- Requires specialized hardware (sensor pen) and integration.
- Data collection and labeling are more involved.
- Sensor calibration and preprocessing add complexity.

Offline (image-based) advantages:
- No special hardware needed — just scans or photos.
- Large mature datasets and well-developed OCR models exist.
- Easier annotation workflows (visual inspection, crowd labeling).

Offline disadvantages:
- Loses temporal dynamics — harder to disambiguate some strokes.
- Sensitive to image quality, lighting, and occlusion.

When to use which:
- Use online/sensor methods when you control acquisition (apps, pens) and need dynamics, real-time feedback, or robustness to scanning issues.
- Use offline OCR when working with scanned documents, historical archives, or images where sensors are unavailable.

---

Q: What are the advantages and disadvantages of OCR (image-based) specifically?

A: Advantages of OCR
- Mature ecosystem: many pre-trained models and tools (Tesseract, CRNNs, commercial APIs).
- Works with existing scanned archives and images.
- High accuracy on clean, well-scanned text and print.

Disadvantages of OCR
- Loses temporal stroke information; handwriting ambiguity can be harder to resolve.
- Performance degrades with poor image quality, complex backgrounds, or unusual handwriting styles.
- May require language models/post-processing to correct errors.

Q: Which model architecture is best suited for sensor pens?

A: There is no single best model; choice depends on the task and data:

- Character-level classification (isolated strokes): 1D-CNNs, small RNNs (LSTM/GRU), or a hybrid CNN+GNN that captures local temporal features and node/temporal relationships. `cnn_gnn.py` is an example of this hybrid idea.
- Sequence recognition (words/equations): sequence-to-sequence encoder-decoder with attention, CTC-based models (Conv + LSTM + CTC), or Transformer-based encoder-decoders. Transformers perform very well with enough data.
- If you have multichannel sensors (13 channels as in OnHW), use an initial CNN trunk or 1D conv layers to learn local features per time step, then a temporal model (RNN/Transformer) or GNN to model relationships.
- If you need confidence estimates or robustness to domain shift, consider deep ensembles or SWAG (uncertainty-aware methods).

Recommendations by scenario:
- Small dataset / quick experiments: 1D-CNN or CNN+LSTM (fast to train).
- Medium to large dataset: Transformers or CNN+Transformer encoder for best accuracy.
- Limited compute: smaller CNN+RNN with careful regularization.

---

Q: How about datasets and how to load them? What format does this repo expect?

A: Dataset notes for this repo:

- `cnn_gnn.py` expects two pickle files in `data/`:
  - `data/all_x_dat_imu.pkl` — list (or array) of N samples where each sample is a 2D numpy array shaped (T_i, C) with T_i time steps and C channels (typically 13 channels for OnHW). Each sample is one character recording.
  - `data/all_gt.pkl` — list of N ground-truth labels (character strings or sequences) aligned with the IMU samples.

Loading example (same approach used in `cnn_gnn.py`):

```python
import pickle

with open('data/all_x_dat_imu.pkl','rb') as f:
    imu_data = pickle.load(f)  # list of numpy arrays, shape (T_i, C)

with open('data/all_gt.pkl','rb') as f:
    gt_data = pickle.load(f)   # list of labels (e.g. 'A', 'B', ...)

# Pad sequences for batch training
from keras.preprocessing.sequence import pad_sequences
imu_data_padded = pad_sequences(imu_data, padding='post', dtype='float32')
```

For sequence datasets (words/equations) the ground truth is a sequence (string) per recording; you will need tokenization and a sequence loss (CTC or cross-entropy for decoder-based models).

Where to get the data
- The OnHW dataset page (Fraunhofer IIS) provides downloads and documentation: https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html

---

Q: What is the recommended RAM / PC for training?

A: Resource needs depend on model size, batch size, and dataset size. Guidelines:

- Small experiments (character classifier; CPU-only): 8–16 GB RAM, modern 4-core CPU — okay for prototyping with small batches.
- Moderate training (CNN+RNN on character datasets): 16–32 GB RAM and a GPU with 6–12 GB VRAM (e.g., NVIDIA RTX 2060/3060) will significantly speed training.
- Larger sequence models or Transformer training: GPU with 12–24 GB VRAM (RTX 3080/3080 Ti, 2080 Ti, or better). 32+ GB system RAM recommended.

Storage: the OnHW dataset downloads are hundreds of MB to a few GB depending on splits — make sure you have 10–50 GB free to be safe.

Summary (recommended machine for comfortable training):
- GPU: NVIDIA RTX 3060 / 3070 or better (12+ GB VRAM preferred)
- CPU: 6–12 cores
- RAM: 32 GB
- Disk: 100 GB (SSD preferred)

If you don't have a GPU, you can still run small experiments on CPU but expect much slower training.

---

Q: Which dataset splits are available (writer-dependent / writer-independent)?

A: Many OnHW datasets provide both writer-dependent and writer-independent splits. Sequence datasets commonly list multiple zip files (Right-handed / Left-handed, writer-dependent / writer-independent). Check the dataset README on the Fraunhofer page for exact file names (the repo's `BIBLIOGRAPHY.bib` points to the relevant papers).

---

Q: Anything missing in the README about sequence vs character datasets?

A: The README has been updated to include a row for the sequence-based OnHW datasets. In short:

- Character datasets: one example per isolated character — use character classifiers.
- Sequence datasets: continuous streams (words/equations) — use seq2seq/CTC/Transformer models.

---

Q: Can you summarize what to use when (practical rule-of-thumb)?

A:
- If your input is isolated characters (one pen stroke per class): use a character model (CNN / CNN+GNN / small RNN).
- If your input is continuous handwriting (sentences/words): use a sequence model (CTC or seq2seq/Transformer).
- If you have both IMU and images and want to leverage both: use cross-modal representation learning (triplet loss or contrastive learning) to align modalities.