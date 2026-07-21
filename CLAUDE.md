# CLAUDE.md — imu2text (IMU handwriting-recognition models)

## Working rules (apply to every change)

- **Honest evaluation is this repo's identity.** Its own README documents
  the lesson: `cnn_gnn.py`'s "~99%" was train-set memorization; the real
  held-out figure was ~43–47%. Never report a number without stating the
  split (writer-independent vs random), and never let train data leak into
  normalization or evaluation. Publicly quoted accuracy is ~65–80% on new
  writers — don't inflate it anywhere.
- **Verify before claiming.** Numbers come from an actual run or a
  committed result in `results/` — never from memory or extrapolation.
  Cite which script and split produced any figure you quote.
- **No AI-isms** in docs, comments, or commit messages; plain, specific
  language.
- **Conventional commits** (`feat:`, `fix:`, `docs:`, `test:`); body says why.
- **Build and test before every commit; CI green before merge.**

## Commands (mirror of `.github/workflows/ci.yml`)

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu  # CPU/CI
python -m py_compile *.py            # syntax gate (CI does this)
pytest                                # tests/ — split, writer inference, augmentation, CTC
python onhw_seq2seq.py --demo         # verifies the seq2seq pipeline on synthetic data, no dataset needed
```

`requirements.txt` pins are **security-motivated** (torch 2.7.1 for
CVE-2025-32434, scikit-learn 1.5.2 for CVE-2024-5206) — don't loosen them
without checking the advisories.

## Repo map

- `onhw_models.py` — the honest benchmark suite: baselines + SOTA
  CNN+BiLSTM, writer-independent and random splits; class set inferred
  from labels (handles OnHW-chars and OnHW-symbols).
- `onhw_seq2seq.py` — CTC seq2seq (words/equations), CER/WER metrics.
- `cnn_gnn.py` — legacy single-script example; keep for reference, don't
  extend, and never quote its self-evaluation numbers.
- `make_learning_curve.py` / `plot_results.py` / `onhw_projection.m` —
  learning curve, figures, logistic projection to full-dataset scale.
- `tests/` + `pytest.ini` (`pythonpath = .` so tests import the top-level
  scripts regardless of invocation).

## Cross-repo contract

- Training data comes from **vahinitech/datasets** (`build_dataset.py` →
  `all_x_dat_imu.pkl`, `all_gt.pkl`, `writers.pkl`, codes in the same
  order). Writer-independent splits depend on that ordering — schema
  changes must be coordinated there, and the collection kit itself lives
  there, not here.
- `writers.pkl` holds pseudonymous student codes only. Nothing in this
  repo may introduce or log real identities; results and figures are
  reported in aggregate (see datasets repo issue #6 for the
  identity-separation rules).
