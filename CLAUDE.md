# CLAUDE.md — imu2text (IMU handwriting-recognition models)

## Working rules (apply to every change)

- **Evaluation rigour is this repo's identity.** Its own README documents
  the lesson: `legacy/cnn_gnn.py` evaluates on its training array, so its
  self-reported accuracy is not a held-out figure. Never report a number
  without stating the
  split (writer-independent vs random), and never let train data leak into
  normalization or evaluation. Publicly quoted accuracy is ~65–80% on new
  writers — don't inflate it anywhere.
- **Verify before claiming.** Numbers come from an actual run or a
  committed result in `results/` — never from memory or extrapolation.
  Cite which script and split produced any figure you quote.
- **Never copy another researcher's code into this repo — reference it,
  don't paste it.** This repo already does this right:
  `docs/impacx_onhw_analysis.md` explicitly analyzes ImpAcX_OnHW's
  pipeline and `scripts/plot_results.py` is rebuilt "in the style of" its
  `plot_kNN_results.py" — independently reimplemented, cited, not copied.
  When a paper's method or a public repo's implementation is genuinely
  needed: (a) cite the paper/repo and reimplement it independently in our
  own code, or (b) if literal reuse is truly unavoidable, get the
  original author's explicit consent first and record who granted it and
  under what terms in the commit/PR before merging. This is a research
  project — unattributed code reuse is an academic-integrity and IP risk,
  not a style nitpick. Applies with extra force to AI-assisted changes: a
  model can reproduce code it saw during training without anyone noticing
  the provenance, so treat any suspiciously polished or unusually-styled
  block as a prompt to check where it actually came from before it ships.
- **No AI-isms** in docs, comments, or commit messages; plain, specific
  language.
- **Conventional commits** (`feat:`, `fix:`, `docs:`, `test:`); body says why.
- **Build and test before every commit; CI green before merge.**
- **Benchmarking has its own rules** - see the `benchmarking` skill
  before running or reporting any accuracy number. The short version: seed
  through `keras.utils.set_random_seed`, use `--deterministic` for any
  before/after comparison, measure the noise floor on the dataset you are
  actually using, and quote the split, its provenance, the dataset and the
  class count with every figure.
- **Docs-only changes skip CI** — `ci.yml` has `paths-ignore: ['**/*.md',
  'docs/**']`; a PR touching only markdown never triggers the pipeline. A
  mixed PR (docs + code) still runs everything.

## Commands (mirror of `.github/workflows/ci.yml`)

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu  # CPU/CI
python -m py_compile *.py            # syntax gate (CI does this)
pytest                                # tests/ — split, writer inference, augmentation, CTC
python -m imu2text.seq2seq --demo         # verifies the seq2seq pipeline on synthetic data, no dataset needed
```

`requirements.txt` pins are **security-motivated** (torch 2.7.1 for
CVE-2025-32434, scikit-learn 1.5.2 for CVE-2024-5206) — don't loosen them
without checking the advisories.

## Repo map

Layout: `imu2text/` is the package (loaders, augmentation, models, seq2seq),
`scripts/` holds standalone runners, `legacy/` holds `cnn_gnn.py`. The package
is imported from the source tree rather than installed - `pytest.ini` puts the
repo root on `sys.path`. Entry points are `python -m imu2text.<module>`.

- `imu2text/models.py` — the benchmark suite: baselines + SOTA
  CNN+BiLSTM, writer-independent and random splits; class set inferred
  from labels (handles OnHW-chars and OnHW-symbols).
- `imu2text/seq2seq.py` — CTC seq2seq (words/equations), CER/WER metrics.
- `legacy/cnn_gnn.py` — legacy single-script example; keep for reference, don't
  extend, and never quote its self-evaluation numbers.
- `scripts/make_learning_curve.py` / `scripts/plot_results.py` / `scripts/onhw_projection.m` —
  learning curve, figures, logistic projection to full-dataset scale.
- `imu2text/chars.py` / `imu2text/symbols.py` / `imu2text/words.py` / `imu2text/download.py` -
  loaders for the published Fraunhofer archives, plus the download catalog.
  Verified against the real ZIPs; `imu2text/augment.py` holds the transform policy.
- `tests/` + `pytest.ini` (`pythonpath = .` so tests import the top-level
  scripts regardless of invocation). `tests/test_real_data.py` runs the
  loaders against real archives and skips unless `ONHW_DATA_DIR` is set -
  synthetic fixtures share the loaders' assumptions and cannot catch a format
  mismatch on their own.

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
