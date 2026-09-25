# imu2text

Handwriting recognition from a sensor pen. imu2text trains on the Fraunhofer
IIS OnHW datasets (13 IMU channels at 100 Hz) and reports accuracy on writers
the model has never seen.

## Quick start

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m imu2text.download onhw_chars --out ./data     # 896 MB, once
python -m imu2text.models --models cnn_bilstm_attn \
    --onhw-chars data/onhw-chars_2021-06-30 --case both --dependency indep --fold 0 \
    --augment 2 --aug-policy extended --label-smoothing 0.1 --lr-schedule --epochs 30
```

No download needed to check the sequence pipeline: `python -m imu2text.seq2seq --demo`.

## Results

Official splits, writer-independent, fold 0, one seed.

| Task | Metric | imu2text | Published |
|---|---|--:|--:|
| OnHW-chars, 52 classes | accuracy | **72.5%** | 68.06% (Ott et al., ACM MM 2022, CNN+BiLSTM) |
| OnHW-Words500, 59 characters | greedy CER, refit on all training writers | **53.95%** | not compared |

On the characters, 43% of the remaining errors are a letter read as its
other case; scored case-insensitively the same model reaches 84.3%.

Details, every other split and the reproduction commands:
[docs/benchmarks.md](docs/benchmarks.md).

## Docs

- [Getting started](docs/getting_started.md): the problem, the code, where to begin
- [Benchmarks](docs/benchmarks.md): all results and how they were measured
- [Datasets](docs/datasets.md): the OnHW archives and how to load them
- [Roadmap](docs/roadmap.md): what is done and what comes next
- [Glossary](docs/glossary.md): the abbreviations

## Contributing

Read [AGENTS.md](AGENTS.md) first. CI runs `pytest`, `black`, `pylint` and a
Markdown style check.

## License and data

The code is Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Maintained by [@vahinitech](https://github.com/vahinitech).

The OnHW datasets are by Fraunhofer IIS, for non-commercial use only. They are
not covered by the Apache license and this repository does not contain them:
`python -m imu2text.download` fetches them from Fraunhofer. If you use them,
cite Ott et al., "The OnHW Dataset: Online Handwriting Recognition from
IMU-Enhanced Ballpoint Pens with Machine Learning", Proc. ACM IMWUT 4(3), 2020.
