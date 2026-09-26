# Vahini playground

One handwritten letter, symbol or word, followed through the pipeline: the
pen's 13 sensor channels, a filter, the trained models, and what they think
was written. Built for people new to the project, and as a map of where to
contribute.

Published at https://playground.vahinitech.com, pinned to one imu2text
commit. Locally, open `index.html` in a browser; there is no server and no
build step.

## Two modes

The page never runs a model. It shows what `scripts/build_playground.py`
exported, so it cannot disagree with the Python code.

**Published page (`data/public.js`, committed).** Model outputs for real test
samples, and one synthetic signal to show what the filters do. No OnHW
recordings. What each task draws on:

| Task | Writers | Model | Saved outputs |
|---|---|---|---|
| Characters | unseen, right- and left-handed | 5-seed ensemble | `results/ensemble/` |
| Characters | seen | one model, seed 0 | `results/tasks/chars_dep.npz` |
| Symbols, equations | unseen and seen | one model, seed 0 | `results/tasks/` |
| Words | unseen | CTC, greedy and word-list decoding | `results/ctc/refit_seed0.json.predictions.npz` |

**Local (`data/local.js`, gitignored).** The right-handed character samples
with their real recordings, raw and filtered. Download the data yourself.

```bash
python -m imu2text.download onhw_chars --out ./data
python -m scripts.build_playground \
    --members 'results/ensemble/right_seed*.npz' \
    --members-left 'results/ensemble/both_seed*.npz' \
    --task chars_dep=results/tasks/chars_dep.npz \
    --task symbols_indep=results/tasks/symbols_indep.npz \
    --task symbols_dep=results/tasks/symbols_dep.npz \
    --task equations_indep=results/tasks/equations_indep.npz \
    --task equations_dep=results/tasks/equations_dep.npz \
    --onhw-chars data/onhw-chars_2021-06-30   # leave out for public.js only
```

The chart sections read the same file. "Compare the algorithms" shows the
published OnHW-chars table (Ott et al., ACM MM 2022, Table 3) next to this
repo's runs. Training curves, calibration and mix-ups per algorithm appear
once each built-in design has a saved run in `results/algorithms/`:

```bash
bash scripts/run_algorithms.sh data/onhw-chars_2021-06-30   # hours on a CPU
```

The OnHW datasets are by Fraunhofer IIS, for non-commercial use, and are not
covered by this repository's Apache-2.0 license. Do not commit `local.js`.

## Add a method

1. Write it in the package (a filter in `imu2text/filters.py`, a model in
   `imu2text/models.py`) with a test, citing the paper it comes from.
2. Measure it on the official split with `--deterministic` and commit the
   result.
3. Add it to `stages.js` with the accuracy, its conditions and the file that
   holds the result. Remove its card from `open` if it was there.
4. Rebuild `data/public.js` if the model outputs changed.

The cards in step 4 of the page are the open tasks. Each links to an issue.

## Files

| File | What |
|---|---|
| `index.html`, `style.css`, `app.js` | the page; plain JavaScript, no dependencies |
| `charts.js` | "Compare the algorithms" and "How sure is the AI?" charts |
| `write.js` | "Draw a letter": your drawing picks a real OnHW recording of that character and shows the model's answer (the drawing and shape guess are simulated) |
| `stages.js` | the methods shown, with measured accuracies and sources |
| `data/public.js` | generated model outputs and the synthetic signal |
| `data/local.js` | generated real recordings, local only |
| `favicon.svg`, `og.png`, `robots.txt`, `sitemap.xml` | icon, social preview, crawler files |
| `og.html` | source of `og.png`; regenerate with the command in its header |

The Vahini design system (colours, type, spacing and shared components),
fonts and logo are not in this repository. Every Vahini host serves them at
the same paths, and the page links them there:

| Path | What |
|---|---|
| `/site/design/v1/vahini.css` | `--v-*` tokens and the shared `v-*` classes |
| `/site/assets/fonts/` | the font files |
| `/site/assets/vahini-logo.png` | the logo; the page shows a drawn mark without it |

`style.css` uses only `var(--v-*)` tokens and holds the layout specific to
this page; `tests/test_playground_page.py` fails on a colour code. To work on
the page outside the deployment, fetch the live stylesheet next to it (the
`site/` folder is gitignored):

```bash
cd playground
curl --create-dirs -o site/design/v1/vahini.css https://vahinitech.com/site/design/v1/vahini.css
python -m http.server 8000
```

The name and logo belong to Vahini Technologies and are not covered by the
Apache-2.0 license.

The search and sharing metadata (title, description, canonical URL, Open
Graph, JSON-LD) is in the head of `index.html`. `tests/test_playground_page.py`
checks it, and that the numbers written into the HTML for crawlers match
`data/public.js`.

A view can be shared with its URL, for example
`index.html#task=symbols&protocol=dep&sample=3&filter=lowpass`.
