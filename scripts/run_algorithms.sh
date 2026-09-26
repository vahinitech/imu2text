#!/usr/bin/env bash
# Train every built-in design once on the same official test, so the
# playground can compare them: training curves, accuracy, calibration and
# confusion per algorithm.
#
# Same split and settings for all: OnHW-chars both/indep/fold0 (52 classes,
# writer-independent), 30 epochs, augmentation x2 (extended), label
# smoothing 0.1, LR schedule, seed 0, --deterministic. One file per design in
# results/algorithms/. Needs the downloaded archive; CPU takes hours.
#
#   python -m imu2text.download onhw_chars --out ./data
#   bash scripts/run_algorithms.sh data/onhw-chars_2021-06-30
#
# Then rebuild the playground data with --algorithms (playground/README.md).
set -euo pipefail

DATA=${1:?usage: scripts/run_algorithms.sh <onhw-chars directory> [design ...]}
shift
DESIGNS=("$@")
if [ ${#DESIGNS[@]} -eq 0 ]; then
    DESIGNS=(cnn lstm bilstm cnn_bilstm cnn_bilstm_attn transformer moe)
fi
mkdir -p results/algorithms

for design in "${DESIGNS[@]}"; do
    out="results/algorithms/${design}.npz"
    if [ -f "$out" ]; then
        echo "skip ${design}: ${out} exists"
        continue
    fi
    echo "== ${design}"
    python -m imu2text.models --models "$design" \
        --onhw-chars "$DATA" --case both --dependency indep --fold 0 \
        --epochs 30 --augment 2 --aug-policy extended \
        --label-smoothing 0.1 --lr-schedule --seed 0 --deterministic \
        --save-predictions "$out" 2>&1 | tee "${TMPDIR:-/tmp}/imu2text-${design}.log"
done
