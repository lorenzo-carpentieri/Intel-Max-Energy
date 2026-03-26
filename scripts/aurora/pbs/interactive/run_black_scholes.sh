#!/bin/bash
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <target_freq> <num_runs>"
  exit 1
fi

TARGET_FREQ="$1"
NUM_RUNS="$2"

echo "Running Black Scholes Throttling Benchmark ..."  # Output: app_name
NNODES=`wc -l < $PBS_NODEFILE`

module restore
module load  oneapi/release/2025.2.0  mpich/opt/develop-git.6037a7a
# module load geopm-runtime/3.2.0
export  ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export SYCL_PI_LEVEL_ZERO_PROFILING=1
export ZE_ENABLE_TRACING_LAYER=1


export SYNERGY_LOG="info"
size=$((8192 * 8192))
BASE_DIR="/home/lcarpent/energy-workspace/Intel-Max-Energy/"
APP_NAME="black-scholes"
CSV_DIR="${BASE_DIR}/logs/app/${APP_NAME}"
mkdir -p ${CSV_DIR}
${BASE_DIR}/bin/black_scholes --csv-path=${CSV_DIR} --runs=${NUM_RUNS} --freq=${TARGET_FREQ} --iters=2048 --local-size=256 --size=${size}
