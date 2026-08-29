---
applyTo: "**"
---

# Code review instructions — vahinitech/imu2text

IMU handwriting-recognition model training code (CNN/GNN/CTC seq2seq).
This repo's own history is the standing lesson for what to watch for:
`cnn_gnn.py`'s "~99% accuracy" was train-set memorization — the honest
held-out figure was ~43–47%.

## Provenance: Never Copied Research Code

- **Flag any newly added code that looks lifted from a paper's official
  implementation or another public repo** (unusual style vs. the rest of
  the file, a comment/variable naming pattern that doesn't match this
  codebase, a suspiciously complete/polished block appearing all at
  once). Ask directly: "is this cited and independently written, or
  copied?"
- The correct pattern already exists here: `docs/impacx_onhw_analysis.md`
  and `plot_results.py` explicitly cite ImpAcX_OnHW/`plot_kNN_results.py`
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
- `cnn_gnn.py` is legacy/reference-only — a PR extending it or citing its
  self-evaluation numbers as current performance should be redirected to
  `onhw_models.py` (the honest benchmark suite) instead.
- Public-facing accuracy claims (docs, PR descriptions, commit messages)
  should match the ~65–80%-on-new-writers figure already established
  elsewhere in the org — flag anything that inflates it.

## Performance Red Flags

- Training loops: watch for unnecessary full-dataset copies, or data
  loading that isn't batched/streamed for larger pickles.
- `onhw_seq2seq.py --demo` must stay lightweight (synthetic data, no
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

## Review Style

- Be specific and cite the function/line.
- No AI-isms in comments, docs, or commit messages.
- Treat any accuracy number in a PR description as a claim that needs
  its split cited before approval, not after.
