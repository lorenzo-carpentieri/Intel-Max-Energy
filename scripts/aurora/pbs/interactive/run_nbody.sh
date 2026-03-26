#!/bin/bash
#PBS -N nbody
#PBS -A  EnergyOpt_PhaseFreq
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=00:20:00
#PBS -l filesystems=home
#PBS -o /home/lcarpent/energy-workspace/Intel-Max-Energy/logs/pbs-out/nbody.out
#PBS -e /home/lcarpent/energy-workspace/Intel-Max-Energy/logs/pbs-out/nbody.err
#PBS -q debug-scaling

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <target_freq> <num_runs>"
  exit 1
fi

TARGET_FREQ="$1"
NUM_RUNS="$2"

echo "Running NBody Benchmark ..."  # Output: app_name
echo "Unique nodes allocated:"
sort -u $PBS_NODEFILE

module restore
module load  oneapi/release/2025.2.0  mpich/opt/develop-git.6037a7a

export  ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
# PROFILING
export ZE_ENABLE_TRACING_LAYER=1
export SYCL_PI_LEVEL_ZERO_PROFILING=1

# export SYNERGY_LOG="debug"
# export SYNERGY_LOG="info"
# MPI example w/ 12 MPI ranks per node each with access to single GPU tile
NNODES=`wc -l < $PBS_NODEFILE`

N_PARTICLES=$((1024*1024))
BASE_DIR="/home/lcarpent/energy-workspace/Intel-Max-Energy/"
APP_NAME="nbody"
CSV_DIR="${BASE_DIR}/logs/app/${APP_NAME}"
mkdir -p ${CSV_DIR}
${BASE_DIR}/bin/nbody --size=${N_PARTICLES} --runs=${NUM_RUNS} --csv-path=${CSV_DIR} --freq=${TARGET_FREQ} --iters=2
