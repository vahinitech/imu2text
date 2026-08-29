# The five OnHW research threads, and why their datasets exist

`docs/onhw_enhancement_guide.md` covers the recognition tasks this repo
already does - characters, symbols, words, equations. This document covers the
five threads *around* those tasks, the ones whose purpose is not obvious from
the dataset name, and proposes an order of work.

Every paper cited here is in `BIBLIOGRAPHY.bib`. Two of them ship as PDFs in
`data/` (`WACV2022_paper.pdf`, `ACMMM_2022.pdf`) and the statements attributed
to those two come from reading them. The other three are cited from their
abstracts and titles only - marked where it matters.

## Why they exist at all

A sensor pen has two problems that a plain "IMU window in, character out"
model cannot address, and both are structural rather than a matter of training
harder:

**The pen does not know where it is.** Accelerometers and gyroscopes measure
motion, not position. Recovering the pen-tip path means integrating
acceleration twice, which compounds drift within a single character. So the
pen cannot draw what was written, and none of the decades of image-based
handwriting recognition applies to its output.

**Labelled paper-and-IMU data is scarce and does not transfer.** OnHW-chars is
31,275 samples from 119 writers, which is small for 52 classes - and a model
trained on it degrades on a different pen, a different writer, or a different
writing surface. Meanwhile tablet handwriting corpora are large and cheap to
label, but a tablet records pen-tip coordinates, not IMU signals.

Each thread attacks one link in that chain:

| Thread | The gap it fills | Dataset it needs |
|---|---|---|
| Pen-tip reconstruction | recover the trajectory the IMU cannot observe | OnHW-wordsTraj |
| Cross-modal representation learning | put IMU and trajectory in one embedding space | wordsTraj (paired IMU + tablet) |
| Tablet/paper domain adaptation | train on plentiful tablet data, deploy on paper | ICROW + OnHW-chars |
| Time-series domain adaptation | the general case: any covariate shift | OnHW-chars WD vs WI, ICROW |
| Uncertainty-aware evaluation | know when the model should not be trusted | any, applied to existing splits |

Read as a programme rather than five separate papers: **reconstruction** gives
you a trajectory, **cross-modal learning** gives you a space where a trajectory
and an IMU signal are comparable, **tablet/paper adaptation** uses that space to
borrow tablet data, **time-series adaptation** generalises the borrowing, and
**uncertainty evaluation** tells you whether to believe any of it. The last one
is what turns a 70%-accurate model into something usable in a classroom: a
recogniser that knows when it is guessing can defer, and one that does not
cannot.

---

## 1. Pen-tip reconstruction and classification

**Paper:** Ott et al., WACV 2022, *Joint Classification and Trajectory
Regression of Online Handwriting using a Multi-Task Learning Approach*
(`data/WACV2022_paper.pdf`).

**Why the dataset is odd.** OnHW-wordsTraj has 16,752 samples from **two
writers** - against 119 writers for OnHW-chars. That looks like a broken
dataset until you see what it costs to make: ground-truth pen-tip position
needs a tablet under the paper *and* cameras on the pen, all synchronised to
the IMU. Two writers is what that rig affords. It is a calibration dataset,
not a training corpus, and it should never be used for a writer-independent
accuracy claim.

**The interesting claim.** The paper's point is that classification and
trajectory regression are *contradictory* objectives - classification wants
features that are invariant to a writer's idiosyncratic geometry, regression
wants exactly that geometry - and that a multi-task architecture with a
distance loss plus a similarity loss improves both anyway.

**What it would take here.** A second regression head on the existing CNN+BiLSTM
trunk emitting (T, 2) pen-tip coordinates, a DTW or soft-DTW alignment loss,
and a loss-weighting scheme. `legacy/cnn_gnn.py` already contains an illustrative
multi-task classification+trajectory example, so the shape is familiar to this
repo - but that script is the legacy one, and its numbers are the memorisation
figures the README warns about, so treat it as a sketch, not a starting point.

**Cost.** 2 GB download (`onhw_wordsTraj_p1`, `onhw_wordsTraj_p2`). New loader,
new metric (trajectory error is not accuracy). The largest single piece of work
of the five.

## 2. Cross-modal representation learning with triplet loss

**Paper:** Ott et al., arXiv 2202.07901, *Cross-Modal Common Representation
Learning with Triplet Loss Functions*. Not read - the following is from the
title, abstract and its role in the series.

**Why it exists.** An IMU sequence and a tablet trajectory of *the same
writing* are two views of one event in completely different units. A triplet
loss learns an embedding where the matched pair lands close together and
mismatched pairs land apart, which makes the two modalities comparable without
either having to be converted into the other.

**Why it is the hinge of the programme.** Once such a space exists, a
classifier trained on tablet embeddings can consume IMU embeddings. That is the
mechanism thread 3 depends on.

**What it would take here.** A two-branch encoder (IMU branch reusing the
existing CNN+BiLSTM trunk, trajectory branch a small 1D-conv net), triplet or
InfoNCE loss, and paired data - which only wordsTraj provides. So it is gated
on thread 1's loader.

**Evaluation caution.** Retrieval metrics on a two-writer dataset will look
excellent and mean very little. Any number here needs the writer split stated,
and with two writers there is no writer-independent split to be had.

## 3. Tablet and paper domain adaptation

**Paper:** Ott et al., MPRSS 2022, *Representation Learning for Tablet and
Paper Domain Adaptation in Favor of Online Handwriting Recognition*. Not read.

**Why it exists - the practical payoff of the whole series.** Tablet
handwriting data is abundant; paper-plus-IMU data is what we are short of. If
the tablet/paper gap can be bridged, the scarce-data problem in the "Why they
exist" section above stops binding.

**The dataset connection worth noticing.** `imu2text/download.py` already lists
`icrow_dep` and `icrow_indep` (103 MB each, URLs verified live). ICROW is
adapted from IRONOFF, a tablet-collected corpus - so **the tablet side of this
experiment is already in our download catalog**, alongside OnHW-chars as the
paper side. This thread needs no new data source, only a loader.

## 4. Domain adaptation for time-series classification

**Paper:** Ott et al., ACMMM 2022 (`data/ACMMM_2022.pdf`).

**What it actually proposes**, from the paper: a two-step *supervised* domain
adaptation. First, find an optimal class-dependent transformation from source
to target domain from a few samples, using optimal-transport methods - earth
mover's distance, Sinkhorn transport, correlation alignment. Second, use
embedding similarity to select which transformation to apply at inference time.

**Why this generalises thread 3.** Tablet-versus-paper is one covariate shift;
this is the method for any of them - a different pen, a different writer
population, a different paper. It is the thread most directly relevant to this
repo's own data, because the Vahini pen has 16 channels against OnHW's 13 and
is a different device: any model trained on OnHW and deployed on Vahini
hardware is exactly the covariate-shift case this paper addresses.

**Cheapest useful experiment.** Correlation alignment (CORAL) is a handful of
lines over the existing features - no optimal-transport solver needed - and the
writer-dependent and writer-independent OnHW-chars folds are a ready-made
source/target pair. This is the best value-per-hour of the five.

## 5. Uncertainty-aware evaluation

**Paper:** Klaß, Lorenz, Lauer-Schmaltz et al., STRL 2022, *Uncertainty-aware
Evaluation of Time-Series Classification for Online Handwriting Recognition
with Domain Shift*. Not read.

**Why it exists, and why it fits this repo particularly well.** This repo's
stated identity is honest evaluation. Accuracy is a single number that hides
whether a model is confidently wrong, and a 70%-accurate recogniser that knows
which 30% it is unsure about is a far more useful product than one that does
not - it can ask rather than guess. The title's "with domain shift" is the
point: calibration degrades under shift faster than accuracy does, so a model
can look fine on accuracy while its confidence becomes meaningless.

**Cheapest of the five by a wide margin.** Expected Calibration Error,
reliability diagrams, and deep-ensemble or MC-dropout uncertainty are all
post-hoc on models this repo already trains. No new dataset, no new
architecture.

---

## Proposed order

Ordered by value per unit of work, not by the order the papers appeared:

1. **Uncertainty-aware evaluation.** No download, no new model. Add ECE and a
   reliability diagram to `imu2text/models.py`'s reporting, and MC-dropout at
   predict time. Directly extends what the repo already claims to care about.
2. **CORAL domain adaptation** on the existing OnHW-chars WD/WI folds. Small,
   self-contained, and it answers a question we actually have about the Vahini
   pen's 16-channel data.
3. **ICROW loader + tablet/paper adaptation.** 103 MB, and the catalog entry
   already exists. Blocked on nothing.
4. **wordsTraj loader + trajectory regression head.** 2 GB and a new metric.
   Real work, but it is the prerequisite for thread 5 and it is the only way
   this pen ever draws what someone wrote.
5. **Cross-modal triplet learning.** Gated on 4's loader.

Two guardrails carried over from `CLAUDE.md` and worth restating because these
threads make them easy to break:

- **wordsTraj has two writers.** No writer-independent claim is possible from
  it. Any number from it is writer-dependent and must say so.
- **Domain adaptation moves the goalposts on what a split means.** A model
  adapted using target-domain samples is not writer-independent in the sense
  the OnHW papers report, even when no target *labels* were used. State what
  the model saw, in the same sentence as the number.

---

## Tracked work

The plan above and the accuracy findings in the README are broken into issues.
Ordered by expected value per unit of work, which is not the order they were
filed:

| Issue | Why it is where it is |
|---|---|
| [#13 uncertainty-aware evaluation](https://github.com/vahinitech/imu2text/issues/13) | No download, no new architecture, and it answers whether the case errors are confidently wrong - which decides whether abstention recovers them in practice |
| [#9 average over all 30 folds](https://github.com/vahinitech/imu2text/issues/9) | Every current number is one seed on one fold; a few hours of CPU removes the caveat from the whole benchmark table |
| [#11 word context for case](https://github.com/vahinitech/imu2text/issues/11) | The largest identified gain. Case is a property of word position, not glyph shape, and the lexicon decoder is already written |
| [#10 factorise letter and case heads](https://github.com/vahinitech/imu2text/issues/10) | Matches the diagnosis directly, and may fail informatively |
| [#14 domain adaptation to the Vahini pen](https://github.com/vahinitech/imu2text/issues/14) | The thread that bears on our own 16-channel hardware |
| [#12 hybrid classical + deep](https://github.com/vahinitech/imu2text/issues/12) | Filed with a prediction of 0 to +2 points, so a null result closes the direction cheaply |
| [#15 sequence truncation study](https://github.com/vahinitech/imu2text/issues/15) | Cheap, and may interact with the case ceiling since capitals run longer |
| [#16 verify the .npy loader contents](https://github.com/vahinitech/imu2text/issues/16) | Four of four loaders disagreed with the published format once already |

The two threads with no issue yet are pen-tip reconstruction and cross-modal
triplet learning. Both need the OnHW-wordsTraj loader, which is 2 GB of
download and a new metric, and neither is worth starting before the cheaper
work above has run.
