# Vahini playground

One handwritten letter, followed through the pipeline: the pen's 13 sensor
channels, a filter, five trained models, and what they think the letter is.
Built for people new to the project, and as a map of where to contribute.

Published at https://playground.vahinitech.com, pinned to one imu2text
commit. Locally, open `index.html` in a browser; there is no server and no
build step.

## Two modes

The page never runs a model. It shows what `scripts/build_playground.py`
exported, so it cannot disagree with the Python code.

**Published page (`data/public.js`, committed).** Model outputs for 24 real
test letters from the official OnHW-chars `both/indep/fold0` split, and one
synthetic signal to show what the filters do. No OnHW recordings.

**Local (`data/local.js`, gitignored).** The same letters with their real
recordings, raw and filtered. Download the data yourself, then:

```bash
python -m imu2text.download onhw_chars --out ./data
python -m scripts.build_playground --members 'results/ensemble/right_seed*.npz' \
    --onhw-chars data/onhw-chars_2021-06-30
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
| `stages.js` | the methods shown, with measured accuracies and sources |
| `data/public.js` | generated model outputs and the synthetic signal |
| `data/local.js` | generated real recordings, local only |

A view can be shared with its URL: `index.html#letter=7&filter=lowpass&model=mean`.
