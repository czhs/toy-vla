#!/bin/bash
# All closed-loop evaluations of the paper. Needs the policy in $TOYVLA_DATA/checkpoints/arm_policy and the folds from fit_fold.py export.
# Usage: arm/run_closed_loop.sh [dev|fresh|big]   (default all). Each process needs 0.9 GiB of GPU memory and up to 8 GB of host RAM; the 9 big shards ran in parallel on one RTX 4090 in 1.1 to 1.7 hours.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; D="${TOYVLA_DATA:-$HERE/../data}"; OUT="$D/arm_eval"; mkdir -p "$OUT"; export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
P="$D/checkpoints/arm_policy"; F="$D/arm_fold"
ev() { python "$HERE/eval_closed_loop.py" --policy smolvla --ckpt "$1" --set "$2" --conds plain --camera "$HERE/assets/camera.json" --lighting shadow --out "$OUT/$3.json"; }
case "${1:-all}" in dev|all)
  ev "$P" dev200 dev_unedited
  for lam in 0.75 1.25 1.5; do ev "$F/fold_A+60_lam$lam" dev200 "dev_fold_lam$lam"; done ;; esac
case "${1:-all}" in fresh|all)
  ev "$P" fresh200 fresh_unedited
  ev "$F/fold_A+60_lam1.25" fresh200 fresh_fold_lam1.25 ;; esac
case "${1:-all}" in big|all)
  for s in a b c; do
    ev "$P" "big3k_$s" "big_unedited_$s" & ev "$F/fold_A+60_lam1.25" "big3k_$s" "big_plus60_$s" & ev "$F/fold_A-60_lam1.25" "big3k_$s" "big_minus60_$s" &
  done; wait ;; esac
