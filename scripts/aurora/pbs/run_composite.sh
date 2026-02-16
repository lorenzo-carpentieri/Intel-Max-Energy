#!/bin/bash
#PBS -N composite_throttling
#PBS -A  EnergyOpt_PhaseFreq
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=00:30:00
#PBS -l filesystems=home
#PBS -o /home/lcarpent/energy-workspace/Intel-Max-Energy/logs/pbs-out/composite_throttling.out
#PBS -e /home/lcarpent/energy-workspace/Intel-Max-Energy/logs/pbs-out/composite_throttling.err
#PBS -q prod

echo "Running Composite Throttling Benchmark ..."  # Output: app_name
echo "Unique nodes allocated:"
sort -u $PBS_NODEFILE
source /home/lcarpent/energy-workspace/Intel-Max-Energy/scripts/aurora/env/set_composite.sh

export SYNERGY_LOG="info"
# MPI example w/ 12 MPI ranks per node each with access to single GPU tile
NNODES=`wc -l < $PBS_NODEFILE`


echo "Exe path: ${EXE_PATH}"
echo "Csv path: ${CSV_FILE}"
echo "Num. benchmark runs: ${RUNS}"
echo "Target freq. tile 0: ${FREQ0}"
echo "Target freq. tile 1: ${FREQ1}"
echo "Kernel tile 0: ${KERNEL1}"
echo "Kernel tile 1: ${KERNEL2}"

UNITRACE_PROF="unitrace --kernel-submission --output-dir-path /home/lcarpent/energy-workspace/Intel-Max-Energy/unitrace-log"
VTUNE_PROF="vtune –collect gpu-hotspots -result-dir=/home/lcarpent/energy-workspace/Intel-Max-Energy/vtune-log "

COMMAND="${EXE_PATH} --csv-path=${CSV_FILE} --runs=${RUNS} --freq0=${FREQ0} --freq1=${FREQ1} --kernel1=${KERNEL1} --kernel2=${KERNEL2}" 
$COMMAND