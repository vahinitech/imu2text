| Group | Test population | n | Members: mean (min-max) % | Ensemble % | Case-insensitive % | ECE % single → ensemble |
|---|---|--:|--:|--:|--:|--:|
| right (5 seeds) | all | 7956 | 72.18 (71.85-72.66) | **74.46** | 86.14 | 2.29 → 4.71 |
| right (5 seeds) | right-handed | 7956 | 72.18 (71.85-72.66) | **74.46** | 86.14 | 2.29 → 4.71 |
| with_left (5 seeds) | all | 8373 | 71.25 (70.44-71.72) | **74.16** | 85.63 | 2.14 → 5.73 |
| with_left (5 seeds) | right-handed | 7956 | 72.16 (71.34-72.67) | **74.92** | 86.55 | 2.28 → 5.66 |
| with_left (5 seeds) | left-handed | 417 | 53.76 (52.28-56.12) | **59.47** | 68.11 | 6.89 → 7.77 |

Splits: right: official both/indep/fold0, writer-independent; with_left: pooled both/indep, right + left. Model: cnn_bilstm_attn.
