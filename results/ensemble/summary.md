| Group | Test population | n | Members: mean (min-max) % | Ensemble % | Case-insensitive % | ECE % single → ensemble |
|---|---|--:|--:|--:|--:|--:|
| right (5 seeds) | all | 7956 | 72.18 (71.85-72.66) | **74.46** | 86.14 | 2.29 → 4.71 |
| right (5 seeds) | right-handed | 7956 | 72.18 (71.85-72.66) | **74.46** | 86.14 | 2.29 → 4.71 |
| with_left (5 seeds) | all | 8373 | 71.25 (70.44-71.72) | **74.16** | 85.63 | 2.14 → 5.73 |
| with_left (5 seeds) | right-handed | 7956 | 72.16 (71.34-72.67) | **74.92** | 86.55 | 2.28 → 5.66 |
| with_left (5 seeds) | left-handed | 417 | 53.76 (52.28-56.12) | **59.47** | 68.11 | 6.89 → 7.77 |

Splits: right: official both/indep/fold0, writer-independent; with_left: pooled both/indep, right + left. Model: cnn_bilstm_attn.

Case errors (a letter read as its own other case) against other errors, whole test set. Single model: mean over the members.

| Group | Model | Kind | n | Median confidence | Confidence ≥ 0.9 |
|---|---|---|--:|--:|--:|
| right | single | right | 5743 | 0.854 | 39.6% |
| right | single | case error | 955 | 0.621 | 4.8% |
| right | single | other error | 1258 | 0.489 | 4.1% |
| right | ensemble | right | 5924 | 0.817 | 33.3% |
| right | ensemble | case error | 929 | 0.584 | 1.9% |
| right | ensemble | other error | 1103 | 0.429 | 1.4% |
| with_left | single | right | 5965 | 0.848 | 38.2% |
| with_left | single | case error | 982 | 0.620 | 4.5% |
| with_left | single | other error | 1426 | 0.469 | 3.2% |
| with_left | ensemble | right | 6209 | 0.803 | 31.0% |
| with_left | ensemble | case error | 961 | 0.585 | 1.2% |
| with_left | ensemble | other error | 1203 | 0.405 | 1.1% |

Abstaining on the least confident predictions (ensemble):

| Group | Coverage | Accuracy kept % | Case errors withheld % | Other errors withheld % | Right answers withheld % | Case share of kept errors % |
|---|--:|--:|--:|--:|--:|--:|
| right | 100% | 74.46 | 0.0 | 0.0 | 0.0 | 45.7 |
| right | 90% | 78.78 | 8.7 | 39.2 | 4.8 | 55.8 |
| right | 80% | 82.51 | 26.2 | 61.3 | 11.3 | 61.6 |
| right | 70% | 85.81 | 46.1 | 73.8 | 19.3 | 63.4 |
| with_left | 100% | 74.16 | 0.0 | 0.0 | 0.0 | 44.4 |
| with_left | 90% | 78.66 | 9.2 | 38.9 | 4.5 | 54.3 |
| with_left | 80% | 82.29 | 23.9 | 62.2 | 11.2 | 61.6 |
| with_left | 70% | 85.63 | 44.0 | 74.7 | 19.2 | 63.9 |

Case errors whose second choice is the right answer (ensemble): right 91.0%; with_left 88.1%.
