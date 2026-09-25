# AGENTS.md

Rules for anyone changing this repo, human or coding agent. Claude Code reads
this file through `CLAUDE.md`, and Copilot review reads
`.github/instructions/`, which points back here. Change the rules here, not in
three places.

imu2text trains handwriting recognisers on the Fraunhofer OnHW datasets (an
IMU pen, 13 channels at 100 Hz) and reports writer-independent accuracy on the
official splits.

## Before you commit

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m py_compile $(git ls-files '*.py')
black --check $(git ls-files '*.py')
pylint --fail-under=9.0 $(git ls-files '*.py' ':!tests/*')
pylint --fail-under=9.0 --rcfile=tests/.pylintrc tests/
pytest
python -m imu2text.seq2seq --demo     # synthetic smoke test, no download
```

CI runs the same checks. `ci.yml` skips a PR that touches only Markdown, so
`docs.yml` runs the Markdown style check (`tests/test_docs_style.py`) on
every PR.

Commits use conventional prefixes (`feat:`, `fix:`, `docs:`, `test:`). The
body says why.

## Numbers

- Every accuracy, CER or WER states its split (writer-independent or
  writer-dependent, published or constructed here), the dataset, the class
  count and the seed. "72.5%" is not a result. "72.5%, official
  `both/indep/fold0`, 52 classes, seed 0" is.
- Numbers come from a run or a file in `results/`, never from memory.
- A before/after comparison uses `--deterministic` and states the noise floor
  measured on the same dataset.
- Nothing fitted may see test data: normalisation, charset, lexicon, early
  stopping, model selection, hyperparameter search, ensemble weights,
  BatchNorm statistics. Validation comes from training writers.
- `legacy/cnn_gnn.py` scores itself on its training array. Never quote it.
- The full rules are in `.claude/skills/benchmarking/SKILL.md`.

## Code from other people

Never paste another researcher's or project's code. Cite the paper and write
the method ourselves. If literal reuse cannot be avoided, get the author's
written consent first and record who gave it, and on what terms, in the PR.
This matters most for AI-written code, which can reproduce code from its
training data without anyone noticing. A polished block in a style unlike
the file around it is a reason to ask where it came from.

Cite published papers only. Do not name, link or copy private or reference
repositories.

## Writing

Docs, comments, commit messages and PR bodies follow
`.claude/skills/natural-writing/SKILL.md`. In short:

- No em dashes, and no spaced hyphen used as a dash. Use a comma, a colon or
  a new sentence.
- None of the words on the skill's kill list. The test checks the worst of
  them.
- Say the number and its conditions; drop the adjectives.
- Sentence-case headings, no emoji, no closing summary.
- A PR body says what changed and what the reviewer must check. Detail goes
  in `docs/`.
- One topic, one doc. Update the existing page instead of adding a new one,
  and link rather than copy a table.

`tests/test_docs_style.py` enforces the dash and word rules.

## Security

- `requirements.txt` pins are security-motivated (`torch==2.13.0`,
  `scikit-learn==1.5.2`; the advisory for each is noted next to the pin).
  Check the advisories before loosening a pin.
- Load untrusted checkpoints with `weights_only=True`. The OnHW archives are
  pickles, so only load files from the catalog in `imu2text/download.py`.
- No absolute paths, usernames or credentials in code, logs or committed
  results.
- `writers.pkl` holds pseudonymous codes only. Nothing may log or store a
  real identity, and results are reported in aggregate.
- Do not commit papers, datasets or model weights. Cite papers, download data
  through `imu2text/download.py`, and publish weights outside git.

## Where things are

| Path | What |
|---|---|
| `imu2text/models.py` | character classification: baselines, CNN+BiLSTM, official splits |
| `imu2text/seq2seq.py` | CTC sequence recognition for words and equations, CER/WER |
| `imu2text/chars.py`, `symbols.py`, `words.py` | loaders for the published OnHW archives |
| `imu2text/download.py` | archive catalog and downloader |
| `imu2text/augment.py` | augmentation policies |
| `scripts/` | benchmark, ensemble and figure scripts |
| `legacy/cnn_gnn.py` | old single-script example; do not extend |
| `results/` | committed predictions, tables and figures |
| `docs/` | benchmarks, datasets, roadmap, root-cause analyses |
| `tests/test_real_data.py` | loader tests on the real archives; needs `ONHW_DATA_DIR` |

The package runs from the source tree. `pytest.ini` puts the repo root on
`sys.path`; run modules as `python -m imu2text.<module>`.

## Mistakes this repo has already made

- **Loaders tested only against their own fixtures.** Four loaders passed
  their tests and could not open the real archives, because each fixture
  shared the loader's assumptions. A loader change needs a test in
  `tests/test_real_data.py`.
- **Missing metadata filled with a plausible value.** Absent writer IDs
  became zeros, which reads downstream as "one writer". Use
  `imu2text.chars.WRITER_UNKNOWN` and reject it.
- **Seeding that did not seed.** `tf.random.set_seed` does not reach the
  Keras initialisers; use `keras.utils.set_random_seed`.
- **A split rebuilt from an archive that ships one.** Use the published
  split. The OnHW symbols `dep` archive shares all 27 writers between train
  and val by design.
- **More capacity for a data limit.** On the 52-class task, 43% of the 72.5%
  model's errors are a letter read as its other case (38.4% for the 68.0%
  baseline), and the case-insensitive score of the 72.5% model is 84.3%.
  Extra parameters do not fix that. See `docs/benchmarks.md`.
