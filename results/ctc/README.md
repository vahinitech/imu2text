# CTC accuracy study artifacts

The word study uses the published right-handed OnHW-Words500 writer-independent
fold 0: 19,915 nonempty official training recordings, 5,292 test recordings,
53 writers (11 test writers), and 59 character symbols plus CTC blank.

- `protocol.json`: planned configurations, archive checksums, and excluded runs.
- `environment.json`: CPU, memory, and numerical-library versions.
- `parent_seed0_run*.json`: the parent pipeline at commit `03d1c8f`.
- `fixed_seed0_run*.json`: corrected training with writer-disjoint validation.
- `fixed_random_seed0_run1.json`: corrected training with the parent's inner split.
- `refit_seed0.json`: final fitting on all official training writers, using the
  grouped-validation-selected epoch count.
- `*.json.predictions.npz`: references and predictions used to audit each score;
  corrected selection runs also retain partition indices.
- `summary.json`: recomputed metrics, deterministic repeat checks, duration
  strata, decoder tradeoffs, and paired writer-bootstrap intervals.
- `character_runs.json`, `chars_*.npz`, and `chars_*.txt`: the separate 52-class
  OnHW-chars regression, with its published `both/indep/fold0` partition,
  23,316 nonempty official training recordings and 7,956 test recordings.

Run-time source hashes remain unchanged in the word reports. Syntax fingerprints
allow documentation edits; the `seq2seq.py` library fingerprint also omits its
CLI `main`, which the benchmark and refit bypass. Training-function changes are
rejected. The historical parent implementation itself is loaded from its Git
revision, while the other hashes describe the benchmark harness and dependencies.

The final `.weights.h5` and `.normalization.npz` files are generated locally and
ignored by Git. The report includes their filenames, the weight checksum, and
whether loading the weights into a fresh model reproduced its outputs.

See the [benchmark table and reproduction commands](../../docs/benchmarks.md)
and the [root-cause analysis](../../docs/rca_ctc_lengths.md). The word measurements
are single-fold, single-seed CPU experiments; deterministic repeats measure
repeatability, not variation across seeds or writers outside this fold.
