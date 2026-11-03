#!/bin/bash
# Run every toy experiment script for a few scenes on any device, writing to a temporary directory; a check that the code runs from a clean clone, its numbers mean nothing.
# usage: tests/smoke_toy.sh [PYTHON]    needs data/checkpoints/toy_policy_red.pt (scripts/download_checkpoints.py)
cd "$(dirname "$0")/.."; PY=${1:-python}; TMP=$(mktemp -d); mkdir -p "$TMP/data/checkpoints" "$TMP/results"
ln -s "$PWD/data/checkpoints/toy_policy_red.pt" "$TMP/data/checkpoints/toy_policy_red.pt"; ln -s "$PWD/data/checkpoints/toy_policy_red.pt" "$TMP/data/checkpoints/toy_policy_green.pt"
cp results/head_ranking_top16.json "$TMP/results/"
export TOYVLA_SMOKE=1 TOYVLA_DATA="$TMP/data" TOYVLA_RESULTS="$TMP/results" TOKENIZERS_PARALLELISM=false; fail=0
for s in gradient_directions even_layer_skip keep_erase block_calibration distractor_one_square colour_calibration colour_switch "colour_native_reference green" estimate_gate_scan head_gating query_replacement weight_fold pointer_layer_sweep pointer_test stream_gradients stream_gradient_overlap; do
  t0=$SECONDS; set -- $s; if "$PY" "experiments/$1.py" ${2:+$2} > "$TMP/$1.log" 2>&1; then echo "ok   $s ($((SECONDS - t0)) s)"; else echo "FAIL $s ($((SECONDS - t0)) s)"; tail -15 "$TMP/$1.log"; fail=1; fi
done
echo "outputs in $TMP"; exit $fail
