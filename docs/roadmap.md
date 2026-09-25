# Roadmap for the OnHW dataset family

This repo started as single-character classification on OnHW-chars_L. It
now covers characters, symbols, split equations and OnHW-words500 on the
published splits of the [Fraunhofer IIS OnHW dataset family](https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html).
This file lists what is built, what comes next and in what order, and the
five research threads around the recognition tasks. Measured numbers live in
[benchmarks.md](benchmarks.md); this file only points at them.

## The datasets

All six are recorded with the same sensor pen: two 3-axis accelerometers, a
3-axis gyroscope, a 3-axis magnetometer and a force sensor, the 13 channels
this repo uses, at about 100 Hz. Each ships official 5-fold writer-dependent
(WD) and writer-independent (WI) splits. The OnHW-chars row is checked
against the IMWUT 2020 paper. The other five come from published summaries
of the IJDAR 2022 benchmark paper, which this repo does not hold.

| Dataset | Task | Content | Scale |
|---|---|---|---|
| OnHW-chars | classification (52 / 26 classes) | single characters A–Z, a–z; upper / lower / combined variants | 31,275 samples, 119 writers |
| OnHW-symbols | classification (15 classes) | digits 0–9 and operators (+, −, ·, :, =) | 2,326 samples, 27 writers |
| OnHW-equations | seq2seq | equation strings from the symbols charset | 10,713 samples, 55 writers |
| OnHW-words500 | seq2seq (closed 500-word vocab) | 500 unique words, repeated | 25,218 samples, 53 writers |
| OnHW-wordsRandom | seq2seq (open vocab) | randomly drawn words | 14,641 samples, 54 writers |
| OnHW-wordsTraj | seq2seq + trajectory | words with tablet and camera pen-tip trajectory ground truth | 16,752 samples, 2 writers |

## Done

- Official OnHW-chars benchmark, all six right-handed splits, fold 0: `imu2text/chars.py`, `scripts/make_comparison_table.py`, results in benchmarks.md.
- More writers: the full 119-writer set replaced the 27-training-writer subset whose learning curve (`results/learning_curve.png`) was still climbing.
- Augmentation: `--augment 4 --rnn-units 100 --rnn-layers 2` took OnHW-chars_L from 64.8% to 71.6% on a split that shared writers between train and test (see [benchmarks](benchmarks.md#onhw-chars_l-with-guessed-writers-writer-dependent)), so not WI; rotation, channel dropout and crop are in `imu2text/augment.py` behind `--aug-policy extended`.
- Normalization modes: `--norm per_sample` and the transductive `--norm per_writer` in `imu2text/models.py`; neither helped consistently on OnHW-chars_L.
- Attention pooling, label smoothing and LR schedule: `--models cnn_bilstm_attn`, `--label-smoothing`, `--lr-schedule`.
- Transformer and mixture-of-experts encoders: `build_transformer` and `build_moe` in `imu2text/models.py`, not benchmarked.
- Error analysis: `--error-analysis` and `scripts/plot_error_analysis.py` located the 52-class ceiling in upper/lower case confusions.
- Symbols and split equations: `imu2text/symbols.py` loaders, benchmarked WD and WI.
- Transfer from chars to symbols: `build_transfer_model` and `unfreeze_trunk` in `imu2text/symbols.py`, not yet measured.
- Words500 loader and lexicon-constrained beam search: `imu2text/words.py` (`LexiconDecoder`), used by `imu2text/seq2seq.py --lexicon`.
- CTC alignment and writer-disjoint validation: `imu2text/sequence_data.py`, `scripts/benchmark_ctc.py`, `scripts/refit_ctc.py`, see [rca_ctc_lengths.md](rca_ctc_lengths.md).
- Left-handed chars: `--onhw-chars-l` and `--both-hands` in `imu2text/models.py`, constructed splits only.
- Seed ensembles with ECE, reliability and coverage plots: `scripts/ensemble_chars.py`, `scripts/plot_uncertainty.py`. This is the code for #13; no result is committed yet.
- Dataset downloads, including the ICROW and wordsTraj archives: `imu2text/download.py`.

## Next, in order

Ordered by expected value per unit of work, which is not the order the issues
were filed:

| Issue | Why it is where it is |
|---|---|
| [#13 uncertainty-aware evaluation](https://github.com/vahinitech/imu2text/issues/13) | No download, no new architecture, and it answers whether the case errors are confidently wrong, which decides whether abstention recovers them in practice |
| [#9 average over all 30 folds](https://github.com/vahinitech/imu2text/issues/9) | Every current number is one seed on one fold. The 30 directories are 3 case settings × 2 protocols × 5 folds, so each reported task (e.g. `both/indep`) is averaged over its own 5 folds, never across all 30; a few hours of CPU removes the caveat from the whole benchmark table |
| [#11 word context for case](https://github.com/vahinitech/imu2text/issues/11) | The largest identified gain. Case is a property of word position, not glyph shape, and the lexicon decoder is already written |
| [#10 factorise letter and case heads](https://github.com/vahinitech/imu2text/issues/10) | Matches the diagnosis directly, and may fail informatively |
| [#12 hybrid classical + deep](https://github.com/vahinitech/imu2text/issues/12) | Filed with a prediction of 0 to +2 points, so a null result closes the direction cheaply |
| [#15 sequence truncation study](https://github.com/vahinitech/imu2text/issues/15) | Cheap, and may interact with the case ceiling since capitals run longer |
| [#16 verify the .npy loader contents](https://github.com/vahinitech/imu2text/issues/16) | Four of four loaders disagreed with the published format once already |

After those, untracked and roughly in order:

1. CORAL domain adaptation on the OnHW-chars WD/WI folds (thread 4).
2. ICROW loader and tablet/paper adaptation (thread 3). 103 MB, catalog entry exists.
3. Measure the chars-to-symbols transfer path. Symbols is 6.7 points behind the published WI figure, and regularisation alone did not close it.
4. A DTW-kNN reference baseline for symbols in the style of [ImpAcX_OnHW](https://github.com/KorayKarabina/ImpAcX_OnHW) (see [impacx_onhw_analysis.md](impacx_onhw_analysis.md)), reimplemented, not copied.
5. Loaders for OnHW-equations as sequences and OnHW-wordsRandom, then CTC runs on their official folds. For equations, a digit/operator grammar can prune impossible decodes the way the lexicon does for words500.
6. wordsTraj loader and trajectory regression head (thread 1), then cross-modal triplet learning (thread 2).

For seq2seq, the reference point is REWI ([code](https://github.com/jindongli24/REWI); Li et al., iWOAR 2025): CNN+BiLSTM and modern encoder ablations (ResNet, MLP-Mixer, ViT, ConvNeXt, SwinV2) with CTC, best reported CER 7.37% / WER 15.12% on OnHW-words500 (WI, right-handed) at only ~3.9M parameters. Their small model says capacity is not the bottleneck; writer variation is. The IJDAR 2022 benchmark found attention decoders only beat CTC with much more data, so keep CTC. Ott et al. (2022) also trained conditional GANs on OnHW-symbols to synthesize IMU signals, which is an option if transfer does not help.

## Guardrails

The general rules are in `CLAUDE.md` and enforced in `imu2text/models.py`:
WI first, scaler fit on train only, augment train only, trivial baseline next
to every result, CER/WER for sequences (`imu2text.seq2seq.cer` / `wer`). Two
more that the threads below make easy to break:

- **wordsTraj has two writers.** No writer-independent claim is possible from it. Any number from it is writer-dependent and must say so.
- **Domain adaptation moves the goalposts on what a split means.** A model adapted using target-domain samples is not writer-independent in the sense the OnHW papers report, even when no target *labels* were used. State what the model saw, in the same sentence as the number.

## Research threads

Every paper here is in `BIBLIOGRAPHY.bib`. I read the WACV 2022 and ACM MM
2022 papers in full; the other three are cited from title and abstract only,
marked below.

A sensor pen has two problems that a plain "IMU window in, character out"
model cannot fix by training harder. It does not know where it is:
accelerometers and gyroscopes measure motion, and integrating acceleration
twice drifts within a single character, so the pen cannot draw what was
written. And labelled paper-and-IMU data is scarce. OnHW-chars is 31,275
samples from 119 writers, small for 52 classes, and a model trained on it
degrades on a different pen, writer or surface. Tablet corpora are large and
cheap, but record coordinates, not IMU signals.

| Thread | The gap it fills | Dataset it needs |
|---|---|---|
| Pen-tip reconstruction | recover the trajectory the IMU cannot observe | OnHW-wordsTraj |
| Cross-modal representation learning | put IMU and trajectory in one embedding space | wordsTraj (paired IMU + tablet) |
| Tablet/paper domain adaptation | train on plentiful tablet data, deploy on paper | ICROW + OnHW-chars |
| Time-series domain adaptation | the general case: any covariate shift | OnHW-chars WD vs WI, ICROW |
| Uncertainty-aware evaluation | know when the model should not be trusted | any, applied to existing splits |

Read together: reconstruction gives a trajectory, cross-modal learning makes
a trajectory and an IMU signal comparable, tablet/paper adaptation uses that
to borrow tablet data, time-series adaptation generalises the borrowing, and
uncertainty evaluation says whether to believe any of it. I think the last one
matters most for a classroom: a recogniser that knows when it is guessing can
ask instead.

### 1. Pen-tip reconstruction

Ott et al., *Joint Classification and Trajectory Regression of Online
Handwriting using a Multi-Task Learning Approach*, WACV 2022. OnHW-wordsTraj
has 16,752 samples from two writers because ground truth needs a tablet under
the paper and cameras on the pen, synchronised to the IMU. It is a calibration
set, not a training corpus. The paper argues classification and trajectory
regression pull in opposite directions (writer-invariant features against
writer geometry) and that a multi-task model with a distance loss plus a
similarity loss improves both anyway. Here it means a (T, 2) regression head
on the CNN+BiLSTM trunk, a DTW or soft-DTW loss, and a new metric.
`legacy/cnn_gnn.py` sketches the shape, but its numbers are memorisation
figures; don't start from it. 2 GB download (`onhw_wordsTraj_p1`,
`onhw_wordsTraj_p2`). The largest of the five.

### 2. Cross-modal representation learning

Ott et al., *Cross-Modal Common Representation Learning with Triplet Loss
Functions*, arXiv 2202.07901 (title and abstract only). A triplet loss puts
an IMU sequence and a tablet trajectory of the same writing close together
in one embedding, so a classifier trained on tablet embeddings can consume
IMU embeddings. That is what thread 3 depends on. It needs a two-branch
encoder and paired data, which only wordsTraj provides, so it waits on thread
1's loader. With two writers there is no WI split, and retrieval numbers will
look better than they are.

### 3. Tablet and paper domain adaptation

Ott et al., *Representation Learning for Tablet and Paper Domain Adaptation in
Favor of Online Handwriting Recognition*, MPRSS 2022 (title and abstract
only). If the tablet/paper gap can be bridged, the scarce-data problem stops
binding. `imu2text/download.py` already lists `icrow_dep` and `icrow_indep`
(103 MB each, URLs verified live). ICROW is adapted from IRONOFF, a tablet
corpus, so the tablet side is already in the catalog and only a loader is
missing.

### 4. Domain adaptation for time-series classification

Ott et al., *Domain Adaptation for Time-Series Classification to Mitigate
Covariate Shift*, ACM MM 2022. A two-step supervised method: find an optimal
class-dependent transformation from source to target with optimal transport
(earth mover's distance, Sinkhorn, correlation alignment) from a few samples,
then pick a transformation at inference by embedding similarity. It covers
any covariate shift, including a locally collected set on a different device
with a different channel count. CORAL is a few lines over existing features
with no transport solver, and the OnHW-chars WD/WI folds are a ready
source/target pair. Its Table 4 is the right-to-left-handed adaptation
benchmark that benchmarks.md says this repo has no number for yet.

### 5. Uncertainty-aware evaluation

Klaß, Lorenz, Lauer-Schmaltz et al., *Uncertainty-aware Evaluation of
Time-Series Classification for Online Handwriting Recognition with Domain
Shift*, STRL 2022 (title and abstract only). Accuracy hides whether a model is
confidently wrong, and calibration degrades under shift faster than accuracy
does. ECE, reliability diagrams and ensemble uncertainty are post-hoc on
models this repo already trains, and the ensemble scripts compute them. What
#13 still needs is the run: are the case confusions confidently wrong, or low
confidence and so recoverable by abstaining?
