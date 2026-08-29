---
applyTo: "**"
---

# Code review instructions — vahinitech/imu2text

IMU handwriting-recognition model training code (CNN/GNN/CTC seq2seq).
This repo's own history is the standing lesson for what to watch for:
`legacy/cnn_gnn.py`'s "~99% accuracy" was train-set memorization; the held-out
figure was ~43–47%.

## Provenance: Never Copied Research Code

- **Flag any newly added code that looks lifted from a paper's official
  implementation or another public repo** (unusual style vs. the rest of
  the file, a comment/variable naming pattern that doesn't match this
  codebase, a suspiciously complete/polished block appearing all at
  once). Ask directly: "is this cited and independently written, or
  copied?"
- The correct pattern already exists here: `docs/impacx_onhw_analysis.md`
  and `scripts/plot_results.py` explicitly cite ImpAcX_OnHW/`plot_kNN_results.py`
  and are *independently reimplemented*, not pasted. New code referencing
  a paper's method should follow that model — cite it, write it
  ourselves.
- If a PR's description says something like "adapted from <repo>" or
  "based on <paper>'s code," that needs either an independent
  reimplementation in the diff or documented author consent before
  merge — not a copy-paste with a comment crediting the source. This is a
  research project; unattributed/uncredited reuse is an academic-integrity
  and IP risk, not just a style issue.

## Security Critical Issues

- Dependency pins in `requirements.txt` are security-motivated
  (`torch==2.7.1` for CVE-2025-32434 — the `torch.load` RCE;
  `scikit-learn==1.5.2` for CVE-2024-5206). A PR loosening either needs
  to explain why, not just "for a newer feature."
- Any use of `torch.load`/pickle-based loading on data that isn't fully
  trusted (e.g., a checkpoint from an external source) should use
  `weights_only=True` or equivalent — this is exactly the class of bug
  CVE-2025-32434 was.
- No hardcoded absolute paths or credentials for dataset locations.

## Evaluation Honesty (this repo's #1 review criterion)

- **Never approve a PR that reports accuracy/CER/WER without stating the
  split** (writer-independent vs. random) and what data it was measured
  on. This is not a style nitpick — it's the specific mistake this repo's
  own README documents as its founding lesson.
- Check for train/normalization leakage: normalization stats, vocab, or
  any fitted parameter must be computed from train-only data, never from
  val/test.
- `legacy/cnn_gnn.py` is legacy/reference-only — a PR extending it or citing its
  self-evaluation numbers as current performance should be redirected to
  `imu2text/models.py` (the benchmark suite) instead.
- Public-facing accuracy claims (docs, PR descriptions, commit messages)
  should match the ~65–80%-on-new-writers figure already established
  elsewhere in the org — flag anything that inflates it.

## Dataset Loaders (added after PR #8)

- **A synthetic fixture written alongside a loader tests nothing about the
  real format.** In PR #8 all 65 tests passed while four loaders could not
  open the published archives, because each fixture encoded the same
  assumption as the loader it covered. A PR adding or changing a loader needs
  a test against the real archive, skipped by default (see
  `tests/test_real_data.py`, gated on `ONHW_DATA_DIR`). "The tests pass" is
  not evidence a loader works.
- Formats that were assumed wrong once and will be again: labels stored as
  strings rather than ints, fold directories named `0` rather than `fold_0`,
  label sequences right-padded with the blank index, and archives whose
  train/val files carry a different name from their unsplit equivalents.
  Ask where the format came from - a download, or a guess.
- **Never fill missing metadata with a plausible value.** Absent writer IDs
  became zeros in one loader, which reads downstream as "one writer" and
  turns a writer-independent re-split into a leaky one. Use a sentinel
  (`imu2text.chars.WRITER_UNKNOWN`) and make consumers reject it.
- Real recordings can be degenerate. Three of OnHW-chars' 31,275 samples have
  zero timesteps. A loader should drop them and say so, with counts per split,
  because dropping test samples changes the denominator of any accuracy.

## Protocol Provenance

- **If an archive ships a split, use it.** A PR that re-derives a split from
  an archive that has one needs a reason. The OnHW symbols `dep` archive
  deliberately shares all 27 writers across train and val; synthesising a
  writer-disjoint split from it produces a writer-independent number from
  writer-dependent data, which is a mislabelled result, not a better one.
- Any reported number should say which split it came from **and whether that
  split was published or constructed here**. `--onhw-chars` uses the official
  folds; `--split writer` constructs one.

## Reproducibility

- **`--seed` alone does not pin a run.** `tf.random.set_seed` does not reach
  the Keras layer initialisers; seeding goes through
  `keras.utils.set_random_seed`. A PR that adds a new entry point must seed
  the same way or its numbers are not reproducible.
- A before/after comparison needs `--deterministic`, which adds op
  determinism and single-threaded execution. Without it the same config at
  the same seed varied about five points on OnHW-chars_L and about 0.2 points
  on the official OnHW-chars split.
- **Ask for the noise floor before accepting an improvement.** A PR claiming
  a gain smaller than the run-to-run spread on that dataset has not measured
  anything. Sample size decides the spread, so the floor has to come from the
  dataset in question, not from another one.

## Performance Red Flags

- Training loops: watch for unnecessary full-dataset copies, or data
  loading that isn't batched/streamed for larger pickles.
- `python -m imu2text.seq2seq --demo` must stay lightweight (synthetic data, no
  download) — a PR that makes `--demo` require a real dataset breaks its
  purpose as a pipeline smoke check.

## Code Quality Essentials

- CI runs `py_compile` (syntax gate) and `pytest` — both must pass.
- `pytest.ini` pins `pythonpath = .` so tests import top-level scripts
  regardless of invocation — don't add import-path workarounds inside
  test files instead of relying on this.
- New model/pipeline code should have a corresponding test in `tests/`
  (splitting, writer inference, augmentation, or CTC pipeline, matching
  the existing categories).

## Cross-repo Contract

- Training data comes from `vahinitech/datasets` (`all_x_dat_imu.pkl`,
  `all_gt.pkl`, `writers.pkl`, codes in matching order). A PR changing
  how these are consumed should confirm compatibility with that repo's
  `build_dataset.py` output, not assume a schema.
- `writers.pkl` holds pseudonymous codes only — no PR here should log,
  print, or persist anything that could re-associate a code with a real
  identity.

## Accuracy Claims on OnHW-chars

- The 52-class task has a known ceiling that is **not a modelling problem**:
  38.4% of test errors are a letter confused with its own other case, and for
  same-shape pairs (C/c, O/o, S/s, U/u, V/v, W/w, X/x, Z/z, K/k, P/p) the cue
  is close to absent from the IMU signal (AUC 0.541 on acceleration RMS, 0.590
  on duration). Case-insensitive scoring gives 80.3% where plain scoring gives
  68.0%.
- So a PR proposing more capacity to push 52-class accuracy should be asked
  what it expects to fix. Measured on the official split: 2xBiLSTM-100 gained
  +0.4 test for +5 train over 1x64, and adding augmentation on top of that
  capacity was worse than leaving it off. Regularisation helps; parameters do
  not.
- **Check which published table a comparison is against.** Ott et al., ACM MM
  2022 Table 3 (right-handed, six official splits, CNN+BiLSTM combined WI
  68.06%) is comparable to what this repo trains. Table 4 in the same paper
  reaches 100.00 because it measures supervised domain adaptation onto
  left-handed writers, baseline 25.19 - a PR quoting it as a recognition
  target is comparing against the wrong thing.
- `--error-analysis` prints the confusion breakdown. Ask for it in any PR
  claiming an accuracy change on this task.
- **Do not discard a change on single-configuration evidence.** Attention
  pooling scored 69.0 against a 69.2 baseline on its own and would have been
  dropped; combined with augmentation it was worth +0.9, and +3.0 with the
  rest of the levers.

## Review Style

- Be specific and cite the function/line.
- No AI-isms in comments, docs, or commit messages.
- Treat any accuracy number in a PR description as a claim that needs
  its split cited before approval, not after.
