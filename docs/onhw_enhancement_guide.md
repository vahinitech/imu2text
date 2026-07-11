# Enhancing this repo across the OnHW dataset family

How to grow this repository from single-character classification
(`onhw_models.py`) to the full [Fraunhofer IIS OnHW dataset family]
(https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html):
character recognition, **symbol recognition**, and **sequence-to-sequence
(words / equations)** recognition. Recommendations draw on:

- **Ott et al., IMWUT 2020** - the original OnHW-chars dataset paper.
- **Ott et al., IJDAR 2022** - the sequence benchmark (CTC vs. attention
  encoder–decoder across the words/equations datasets).
- **[REWI](https://github.com/jindongli24/REWI)** (Li et al., iWOAR 2025) -
  robust, efficient writer-independent IMU HWR; CNN+BiLSTM and modern encoder
  ablations (ResNet, MLP-Mixer, ViT, ConvNeXt, SwinV2) with CTC; best reported
  CER 7.37% / WER 15.12% on OnHW-words500 (WI, right-handed) at only ~3.9M
  parameters.
- **[A-Pen-Is-All-You-Need](https://github.com/DrumsnChocolate/A-Pen-Is-All-You-Need)**
  - notebook-based exploration of the 2020 OnHW dataset and the 2021 STABILO
  competition data (work in progress; useful mainly as data-wrangling reference).
- **[ImpAcX_OnHW](https://github.com/KorayKarabina/ImpAcX_OnHW)** - classical
  DTW-kNN / feature-kNN baselines (see `docs/impacx_onhw_analysis.md`).

## 1. The OnHW dataset family

All datasets are recorded with a sensor-enhanced ballpoint pen (two 3-axis
accelerometers, 3-axis gyroscope, 3-axis magnetometer, force sensor = the 13
channels this repo already uses, sampled at ~100 Hz). Each dataset ships with
official 5-fold **writer-dependent (WD)** and **writer-independent (WI)**
splits. Approximate sizes (see the dataset page / papers for exact stats):

| Dataset | Task | Content | Scale |
|---|---|---|---|
| OnHW-chars | classification (52 / 26 classes) | single characters A–Z, a–z; upper / lower / combined variants | ≈31k samples, 119 writers |
| OnHW-symbols | classification (~15 classes) | digits 0–9 and operators (+, −, ·, :, =, …) | ≈1k samples, ~52 writers |
| OnHW-equations | seq2seq | equation strings from the symbols charset | ≈10.7k samples, 55 writers |
| OnHW-words500 | seq2seq (closed 500-word vocab) | 500 unique words, repeated | ≈25k samples, ~53 writers |
| OnHW-wordsRandom | seq2seq (open vocab) | randomly drawn words | ≈14.6k samples, ~54 writers |
| OnHW-wordsTraj | seq2seq + trajectory | words with camera-tracked pen-tip trajectory ground truth | ≈4.3k samples, 2 writers |

What this repo covers today:

| Task | Status |
|---|---|
| Character classification | `onhw_models.py` (CNN / LSTM / BiLSTM / CNN+BiLSTM, WI + random splits, augmentation) |
| Symbol classification | **works out of the box** - `onhw_models.py` builds its class set from the labels, so pointing `IMU_FILE`/`GT_FILE` at OnHW-symbols pickles just works |
| Seq2seq (words / equations) | `onhw_seq2seq.py` - CNN+BiLSTM+CTC scaffold with CER/WER metrics, greedy decoding, and a synthetic `--demo` verifying the pipeline |
| Trajectory regression | `cnn_gnn.py` (illustrative multi-task example) |

## 2. Improving character recognition

Ranked by expected value for this repo:

1. **More writers.** The learning curve (`make_learning_curve.py`,
   `results/learning_curve.png`) shows WI accuracy still climbing at 27
   training writers; the full OnHW-chars set (119 writers) is the single
   biggest win. Use the official WI folds once the full dataset is downloaded.
2. **Stronger augmentation (already implemented, tune it).** `--augment 4
   --rnn-units 100 --rnn-layers 2` lifts WI accuracy 64.8→71.6% on the bundled
   subset. Additional transforms worth adding: channel dropout, small 3-D
   rotations of the accelerometer/gyro/magnetometer triads (simulates pen grip
   variation - physically meaningful for IMU data), and random cropping.
3. **Writer normalization / adaptation.** Per-writer statistics (mean/std per
   channel) remove sensor-mount and grip bias. Ott et al. (ACMMM 2022) go
   further with optimal-transport domain adaptation between writers; a cheap
   variant is fine-tuning the trained model on a few enrollment samples of the
   target writer (writer-dependent personalization).
4. **Modern encoders.** REWI's ablation is a ready-made menu: replace the CNN
   trunk with an InceptionTime/ResNet-style 1-D backbone or a small
   transformer; on their benchmark, a SwinV2-style encoder beat plain CNNs.
   Keep the BiLSTM head - every OnHW study finds conv+recurrent beats either
   alone.
5. **Ensembles & uncertainty.** Deep ensembles over seeds add 1–2 points and
   give calibrated confidence (Klaß et al. 2022 use SWAG/ensembles on OnHW for
   domain-shift detection - useful in production to reject garbage strokes).
6. **Cross-modal training.** Ott et al. (2022) align IMU embeddings with
   offline handwriting-image embeddings via triplet loss; needs image
   renderings of trajectories (OnHW-wordsTraj provides paired data).

## 3. Improving sequence-to-sequence (words / equations)

`onhw_seq2seq.py` is the starting point (CNN+BiLSTM+CTC - the IJDAR 2022 and
REWI baseline architecture). Priorities:

1. **Get the real data.** Download OnHW-equations / words500 / wordsRandom
   from the dataset page, convert to the two-pickle convention
   (list of `(T,13)` arrays + list of strings), and run the official 5 folds
   (WD and WI). Report CER/WER per fold - REWI's 7.37% CER on words500-WI-right
   is the number to beat.
2. **Beam search + language model.** The scaffold decodes greedily. For
   OnHW-words500 the vocabulary is *closed* (500 words): constrain beam-search
   decoding to the lexicon, or simply rescore the top beams against the word
   list - typically a large WER drop for near-zero cost. For equations, a
   grammar (digits/operators alternation) prunes impossible strings.
3. **Architecture upgrades**, in order of evidence:
   - wider/deeper conv trunk with stride instead of pooling (keeps time
     resolution controllable against CTC's `input_len ≥ label_len` constraint);
   - transformer encoder over the conv features (A-Pen-Is-All-You-Need's
     direction; keep CTC loss - IJDAR found attention decoders only win with
     much more data);
   - REWI-style efficiency pass: their 3.9M-parameter model shows capacity is
     not the bottleneck - writer robustness is.
4. **Multi-task with trajectory.** OnHW-wordsTraj's camera ground truth lets
   the seq2seq encoder also regress pen-tip position (as `cnn_gnn.py` sketches
   for chars) - auxiliary trajectory loss regularizes the shared encoder.
5. **Joint training across datasets.** Equations and words share the encoder;
   train one model with a merged charset, or pretrain on wordsRandom and
   fine-tune per dataset. Chars pretraining also transfers into the conv trunk.

## 4. Improving symbol recognition

OnHW-symbols is tiny (~1k samples), so the levers are different:

1. **Transfer from chars.** Pretrain CNN+BiLSTM on OnHW-chars (31k samples),
   replace the softmax with the ~15-class head, fine-tune with a low learning
   rate. Expect this to dominate training from scratch.
2. **Aggressive augmentation.** The existing `--augment` machinery applies
   unchanged (the class set is inferred from labels); at this data scale use
   ×8–×16.
3. **Synthetic data.** Ott et al. (2022) trained conditional GANs specifically
   on OnHW-symbols to synthesize IMU signals; a simpler modern take is motif
   mixing/concatenation (see `make_demo_data` in `onhw_seq2seq.py`) or
   variational autoencoders per class.
4. **Classical baseline.** At ~1k samples, ImpAcX-style DTW-kNN is competitive
   and needs no training - run it as the honesty baseline next to the nets.

## 5. Methodology guardrails (apply to every task)

These are the rules already enforced in `onhw_models.py`; keep them for every
new dataset:

- **Writer-independent first.** WD numbers flatter; the OnHW literature's
  headline numbers are WI. Never let one writer's samples cross splits.
- **Train-only normalization** (scaler fit on train timesteps only).
- **Augment training data only**; val/test stay untouched.
- **Report the majority-class / trivial baseline** next to every result.
- **5 folds, mean ± std** - single-split results on ≤55 writers are noisy
  (ImpAcX's plotting of mean±std clouds across folds is the model to follow;
  `plot_results.py` has the hook).
- **CER/WER for sequences** (`onhw_seq2seq.cer/wer`), accuracy for
  classification.
