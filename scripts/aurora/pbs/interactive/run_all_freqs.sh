#!/bin/bash

set -euo pipefail

DELETE_LOGS=false
if [ "${1:-}" = "--delete-logs" ]; then
  DELETE_LOGS=true
  shift
fi

if [ "$#" -ne 0 ]; then
  echo "Usage: $0 [--delete-logs]"
  exit 1
fi

# TARGET_FREQS=(1550 1500)
TARGET_FREQS=(1450 1400)

NUM_RUNS=3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/home/lcarpent/energy-workspace/Intel-Max-Energy/logs"
SCRIPTS=(
  "run_black_scholes.sh"
  "run_ftle.sh"
  "run_kmeans.sh"
  "run_knn.sh"
  "run_mol_dyn.sh"
  "run_nbody.sh"
  # "run_tc.sh"
)

if [ "${DELETE_LOGS}" = true ]; then
  echo "Deleting logs in ${LOG_DIR}"
  rm -rf "${LOG_DIR}"/*
fi

for target_freq in "${TARGET_FREQS[@]}"; do
  echo "Running benchmarks at frequency ${target_freq} with ${NUM_RUNS} runs"
  for script_name in "${SCRIPTS[@]}"; do
    script_path="${SCRIPT_DIR}/${script_name}"
    echo "Launching ${script_name} ${target_freq} ${NUM_RUNS}"
    sh "${script_path}" "${target_freq}" "${NUM_RUNS}"
  done
done
