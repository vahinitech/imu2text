# Vahini dataset collection protocol — designing for 99% deliverable accuracy

How to collect an in-house IMU-pen handwriting dataset that supports 99%
*product* accuracy. The core lesson from the OnHW work and this repo's own
learning curve: **99% is reached by collection design + task framing, not by
model tuning**. Isolated 52-class character recognition on unseen writers caps
out in the 70s–80s no matter how much data exists (case pairs like o/O or s/S
are motion-identical); 99% comes from stacking four things the dataset must be
*designed* to enable:

    word/sequence context  +  lexicon decoding  +  per-user enrollment  +  reject option

Each section below states what to collect and *why that ingredient needs it*.

## 1. Decide the metric before the first recording

Write down the exact accuracy definition the product will report, because it
dictates the content mix:

| Target metric | What must be collected |
|---|---|
| Word accuracy, closed vocabulary | the product's real vocabulary, written by many writers |
| Character accuracy in words (CER) | words/sentences with per-character prompts |
| Isolated character accuracy | alphabets — but this metric cannot reach 99% WI; keep it as a diagnostic, not the headline |
| Accuracy on accepted samples (reject option) | everything above **plus** deliberately messy samples so the confidence model sees hard negatives |

## 2. Writers are the budget unit, not samples

Writer-independent accuracy scales with the number of *writers*, and saturates
in samples-per-writer very quickly. This repo's measured curve
(`results/learning_curve.png`): 5 writers → 5%, 16 → 30%, 27 → 66%, logistic
ceiling ≈ 76–80% for isolated chars. Plan accordingly:

- **Minimum 150–300 writers** for a production WI model (OnHW-chars has 119;
  REWI's strong words results rest on ~50+ writers *plus* sequence context).
- **Diversity beats volume**: recruit across age (children through elderly —
  motor patterns differ), handedness (collect left-handed writers and *label*
  them; OnHW splits right/left), grip styles, writing speeds, cursive vs.
  print, and native script background.
- Per writer, 30–60 minutes of writing is enough (see content mix below).
  Returning sessions on different days are valuable (day-to-day sensor-mount
  and mood variation) — 2 sessions × 25 min beats 1 × 50 min.

## 3. Content mix per writer

Recommended session script (≈45 min, prompted — the writer copies what a
screen shows, so labels are automatic):

1. **Alphabets ×3** (A–Z, a–z, 0–9, punctuation/symbols the product needs)
   — supports the char/symbol diagnostic tasks and per-class balance.
2. **Pangram sentences ×10** ("the quick brown fox…" and local-language
   equivalents) — every letter appears *in word context*, which is what the
   production model actually sees.
3. **Product vocabulary words, ~150–250** sampled so that across the whole
   cohort every vocabulary word is written by ≥25 distinct writers. If the
   vocabulary is open, sample from a frequency-weighted wordlist plus random
   character n-grams for coverage.
4. **Case-pair drill** (o/O, s/S, c/C, x/X, w/W, z/Z, u/U, v/V, p/P, k/K in
   words) — these drive the residual error; oversample them deliberately.
5. **Free writing, 2–3 minutes** (unprompted, transcribed afterwards) — the
   only honest source of natural-speed, natural-sloppiness data; feeds the
   reject-option threshold calibration.
6. **Enrollment simulation**: 1 alphabet + 20 words recorded *last*, tagged
   `enrollment=true` — this is the data slice that lets you measure how much
   per-user fine-tuning gains, before the product ships.

## 4. Hardware and signal discipline

- Match the OnHW channel set so all existing code transfers: 2× 3-axis
  accelerometer, 3-axis gyroscope, 3-axis magnetometer, force sensor =
  **13 channels**, ≥100 Hz, hardware-timestamped.
- **Record raw.** No filtering, no normalization, no resampling at capture
  time — preprocessing belongs in the training pipeline where it can be
  changed (and where train-only fitting prevents leakage).
- Per-pen calibration recording at session start (pen at rest on the table,
  then a fixed calibration gesture) — lets you correct sensor bias per device
  and detect broken channels automatically.
- Log metadata per recording: anonymized writer ID, pen serial, session ID,
  date, handedness, surface (paper type, pad underneath), firmware version.
  Pen serial matters: models can overfit to a *device*, and you want the test
  split to be able to hold out devices too.
- Use the **force channel for segmentation**: pen-down/pen-up transitions
  give free stroke boundaries and free start/end trimming.

## 5. Ground truth

- **Prompted capture = automatic labels.** The display shows the target
  string; the app stores (recording, target, timestamp). Cheap and exact.
- **QC every sample**: automatic checks (sequence length within expected
  band for the prompt length, no saturated/flat channels, force channel shows
  actual pen-downs), then human spot-check ~5% by rendering the recording.
  Mark failures `discard` or `rewrite`; never silently drop (the discard rate
  itself is product intelligence).
- **Trajectory ground truth for a subset** (like OnHW-wordsTraj): record
  ~5% of sessions on a tablet-over-paper rig or with a camera tracking the
  pen tip. This unlocks the multi-task trajectory-regression auxiliary loss
  (`cnn_gnn.py` sketches it) which regularizes the encoder, and gives you
  rendering-based cross-modal training (Ott et al. 2022).

## 6. Splits: freeze them at collection time

- Publish **official 5-fold writer-disjoint (WI) and writer-dependent (WD)
  splits** with the dataset, exactly as OnHW does — every future experiment
  becomes comparable.
- Lock away a **never-touched holdout of ~20 writers** that no one evaluates
  against until release candidates. Development overfits to any test set it
  can see repeatedly.
- Split by writer AND check pen-serial balance across folds; if a device is
  concentrated in one fold, WI numbers silently become device numbers.
- Store recordings in collection order with session IDs so nothing like the
  `infer_writer_ids` reconstruction heuristic is ever needed again.

## 7. Consent and privacy

Handwriting dynamics are personal, potentially biometric, data. Collect
written consent covering model training and (if intended) dataset
publication; store the writer-ID↔identity mapping separately from the
dataset; check the applicable regime (GDPR: likely personal data, possibly
special-category if used for identification). Children's data needs guardian
consent. Decide *before* collection whether the dataset may ever be released
publicly — retrofitting consent is impossible.

## 8. Storage format

Keep this repo's two-file convention per subset so everything runs unchanged
(`list of (T,13) float32 arrays` + `list of label strings`), plus a metadata
table (CSV/parquet: recording ID, writer, pen, session, handedness, split
assignments, QC status). REWI's MSCOCO-like layout is a good alternative if
you want interoperability with their tooling. Version releases (`vahini-v1`,
`v1.1`…) and never mutate a released version.

## 9. How the 99% is assembled from this dataset

| Stage | Ingredient from the collection | Expected level |
|---|---|---|
| WI CNN+BiLSTM+CTC on words | writers ≥150, word content, pangrams | ~92–94% char-level (REWI: 92.6% at ~50 writers) |
| + lexicon/beam decoding | closed product vocabulary written by ≥25 writers/word | word accuracy high-90s |
| + per-user enrollment fine-tuning | the `enrollment=true` slice proves the gain pre-launch | +1–3 points on that user |
| + reject option (confidence threshold) | free-writing + messy samples to calibrate | **≥99% on accepted samples** at 90–95% coverage |

The last row is the honest formulation of "99%": the system answers when
confident, asks for a rewrite when not, and the dataset above is what makes
both the confidence model and the measurement trustworthy.

## 10. Pilot before scaling

Run a 10-writer pilot first. It will surface prompt-app bugs, calibration
drift, QC blind spots, and session-length problems at 5% of the cost. Train
on the pilot with `onhw_models.py` / `onhw_seq2seq.py`, extend
`make_learning_curve.py`'s curve with your own points, refit the logistic
projection (`plot_results.py`), and only then commit the full recruitment
budget — the refreshed curve tells you how many writers the target needs.
