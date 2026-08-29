# ImpAcX_OnHW analysis: `DTW_KNN.py` and `plot_kNN_results.py`

Analysis of [KorayKarabina/ImpAcX_OnHW](https://github.com/KorayKarabina/ImpAcX_OnHW),
and how this repository rebuilds its matplotlib figures (`scripts/plot_results.py`).

## What the repo does

ImpAcX_OnHW evaluates **classical (non-deep) time-series classifiers** on the
OnHW-chars dataset: k-nearest-neighbours over Dynamic Time Warping (DTW)
distances, and kNN over engineered features (with `tsfresh`-style feature
significance filtering and NCA dimensionality reduction). It is a useful
counterpoint to neural pipelines like ours: no training beyond distance
computation, strong interpretability, but heavy inference cost (DTW-kNN
compares every test sample against every training sample).

## `DTW_KNN.py` - DTW-based kNN classification

Pipeline:

1. **Load** the official OnHW-chars folds (the dataset ships 5
   train/test folds for both writer-dependent and writer-independent splits).
2. **Preprocess** with min–max scaling per sample.
3. **Distance**: three DTW variants over the 13-channel multivariate series:
   - `DTW` - `dtaidistance.dtw_ndim.distance_fast(a, b)` (multivariate exact
     DTW, C implementation);
   - `DTW2` - `tslearn` soft-DTW (`soft_dtw(a, b, gamma=1)`) mapped through an
     exponential to behave like a similarity kernel;
   - `DTW3` - exact DTW rescaled with `np.exp(-x)`.
4. **kNN**: `get_neighbors()` ranks training samples by distance to each test
   sample; `predict_classification()` takes the top-k labels and `get_max()`
   breaks ties by recursively re-voting with k−1 neighbours.
5. **Persist**: per-test-sample neighbour distances go to CSV (so the
   expensive distance matrix is computed once), and accuracies are appended to
   a results CSV with columns `case, dependency, fold, model, accuracy`
   (case = lower/upper/combined; dependency = writer-dependent/independent).

Design notes worth copying: caching the distance matrix, evaluating all 5
official folds, and reporting WD and WI separately. Weaknesses: `O(N_train ·
N_test)` DTW cost, and simple min–max per-sample scaling (our pipeline's
train-fit standardization is leak-free and works better for neural nets).

## `plot_kNN_results.py` - how the figures are built

The script aggregates pickled per-fold score lists and draws
"accuracy vs. k" curves, one line per hyperparameter level:

1. **`set_size(width, fraction)`** converts a LaTeX document width **in
   points** into a matplotlib figure size **in inches**, with height set by
   the golden ratio ((√5−1)/2). This is the standard trick (from
   [jwalton.info](https://jwalton.info/Embed-Publication-Matplotlib-Latex/))
   for publication figures: the figure is created at exactly `\textwidth`, so
   LaTeX never rescales it and fonts stay at their true size.
2. **`get_acc_scores(input_str, n_list, n_id)`** loads
   `output/ml_results/{kNN|NCA_kNN}_fold{0..4}_{nsig|ncomp}{n}.txt` (pickled
   lists of accuracy-per-k, k = 1..49) and reduces the 5 folds to a
   **mean curve** and a **std curve** per hyperparameter level `n`.
3. **`plot_acc(...)`** plots one mean line per level, colour per level from a
   hand-picked list (`'plum', 'gold', 'fuchsia', ...`). A commented-out
   `plt.fill_between(mean ± std)` draws the "cloud" of cross-validation
   variance.
4. Two figures are produced and saved as **PDF with
   `bbox_inches='tight'`**: kNN accuracy vs. k for each `n_significant`
   (feature-count) level, and NCA+kNN accuracy vs. k for each `n_components`
   level. Axis labels report "Average Testing Accuracy (5-fold Cross
   Validation)"; legend sits lower right.

### How we rebuilt this style (`scripts/plot_results.py`)

`scripts/plot_results.py` applies the same mechanics to this repository's results:

| ImpAcX original | This repo |
|---|---|
| `set_size()` pt→inch golden-ratio sizing | same function, same reference |
| per-fold pickles → mean/std curves | `results/learning_curve.csv` (+ optional `results/benchmarks.csv`) |
| accuracy vs. k in kNN | WI accuracy vs. number of training writers, with the logistic projection fit from `scripts/onhw_projection.m` overlaid |
| second figure (NCA+kNN comparison) | model-comparison bar chart (CNN+BiLSTM / BiLSTM / CNN / LSTM / majority) |
| PDF + `bbox_inches='tight'` | same, plus a PNG for quick viewing |
| 10 ad-hoc named CSS colours | a small fixed colourblind-safe palette; every series also direct-labeled, so identity never relies on colour alone |

Run it with:

```bash
python scripts/make_learning_curve.py   # produces results/learning_curve.csv (skip if present)
python scripts/plot_results.py          # writes results/learning_curve.pdf/.png, results/model_benchmarks.pdf/.png
```

The mean±std "cloud" idiom (`fill_between`) becomes relevant here once
`scripts/make_learning_curve.py` is run with multiple seeds; the plotting hook is the
same one ImpAcX left commented out.
