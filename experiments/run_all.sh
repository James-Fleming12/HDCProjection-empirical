#!/bin/bash
# Run all experiments sequentially; skip any whose result CSV already exists.
# Usage: bash experiments/run_all.sh [--force]
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
export PYTHONUNBUFFERED=1
FORCE="${1:-}"

step () {
  local name="$1" csv="$2"; shift 2
  if [ -f "$csv" ] && [ "$FORCE" != "--force" ]; then
    echo "=== $name SKIP (results/$csv exists)"
    return 0
  fi
  echo "=== $name start $(date +%T)"
  local t0=$SECONDS
  if python "$@" > "logs/$name.log" 2>&1; then
    echo "=== $name OK $(date +%T) ($((SECONDS-t0))s)"
  else
    echo "=== $name FAIL $(date +%T) ($((SECONDS-t0))s): see logs/$name.log"
    tail -n 15 "logs/$name.log"
  fi
}

step exp1 results/exp1_projection_sweep.csv experiments/exp1_projection_sweep.py --seeds 3
step exp2 results/exp2_dimension_sweep.csv experiments/exp2_dimension_sweep.py --seeds 3
step exp3 results/exp3_robustness.csv experiments/exp3_robustness.py --seeds 3
step exp4 results/exp4_quantization.csv experiments/exp4_quantization.py --seeds 3
step exp5 results/exp5_extractor.csv experiments/exp5_extractor.py --seeds 3
step exp6 results/exp6_processing.csv experiments/exp6_processing.py --seeds 3
step exp6b results/exp6b_sequence.csv experiments/exp6b_sequence.py --seeds 3
step exp7 results/exp7_learned.csv experiments/exp7_learned.py --seeds 3
step exp8 results/exp8_jl.csv experiments/exp8_geometry.py --seeds 3
step exp9 results/exp9_interactions.csv experiments/exp9_interactions.py --seeds 3

echo "RUNNER DONE $(date +%T)"
