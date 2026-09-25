---
applyTo: "**"
---

# Code review: imu2text

The rules live in `AGENTS.md`. This file lists what a reviewer checks. Cite
the file and line for every comment.

## Provenance

- Does new code look lifted from a paper's official code or another repo? A
  style unlike the rest of the file, a complete polished block, or a comment
  like "adapted from" all call for the question: is this cited and written
  here, or copied? Copying needs the author's written consent recorded in
  the PR.
- No private or reference repository is named, linked or copied.

## Numbers

- Every accuracy, CER or WER in the diff, docs or PR body names its split
  (writer-independent or writer-dependent, published or constructed), the
  dataset, the class count and the seed, and comes from a run or a file in
  `results/`.
- Nothing fitted sees test data: normalisation, charset, lexicon, early
  stopping, model selection, hyperparameter search, ensemble weights,
  BatchNorm statistics.
- A claimed improvement is compared with the noise floor measured on the same
  dataset, using `--deterministic` runs. A gain inside the noise is not a gain.
- If an archive ships a split, the PR uses it. The OnHW symbols `dep` archive
  shares its 27 writers between train and val on purpose.
- Comparisons against Ott et al., ACM MM 2022 use Table 3 (right-handed,
  CNN+BiLSTM combined WI 68.06%). Table 4 measures supervised domain
  adaptation onto left-handed writers and is the wrong target.
- On OnHW-chars 52-class, most remaining errors are case confusions. A PR
  adding capacity to raise that number should say what it expects to fix and
  include `--error-analysis` output.
- `legacy/cnn_gnn.py` is not extended and its numbers are never quoted.

## Loaders and data

- A loader change comes with a test against the real archive in
  `tests/test_real_data.py` (skipped unless `ONHW_DATA_DIR` is set). Passing
  synthetic fixtures proves nothing about the real format.
- Missing metadata is never filled with a plausible value; writer IDs use
  `imu2text.chars.WRITER_UNKNOWN`.
- Degenerate recordings are dropped with per-split counts printed.
- Nothing logs or stores a real identity; `writers.pkl` holds pseudonymous
  codes only.

## Reproducibility and security

- New entry points seed through `keras.utils.set_random_seed`.
- `requirements.txt` pins are not loosened without checking the advisories
  noted next to them.
- Untrusted checkpoints load with `weights_only=True`. No absolute paths,
  usernames or credentials in code or committed results.
- `python -m imu2text.seq2seq --demo` still runs on synthetic data without a
  download.

## Code quality

- CI passes: `py_compile`, `black`, `pylint` (9.0 floor), `pytest`, and the
  Markdown style check.
- New model or pipeline code has a test in `tests/`.
- Tests rely on `pytest.ini` for imports; no `sys.path` hacks in test files.
