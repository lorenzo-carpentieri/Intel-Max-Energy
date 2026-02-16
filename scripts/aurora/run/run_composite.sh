
BASE_DIR=$(pwd)
exe_path="${BASE_DIR}/bin/composite_throttling"
csv_file="${BASE_DIR}/logs/throttling/"
runs=3
freq0=1600
freq1=1500
kernel1=comp
kernel2=none
pbs_path="${BASE_DIR}/scripts/aurora/pbs/run_composite.sh"
################# Test 1: Tile 0 -> freq 1600, Tile 1 -> freq 1200, with same workload #############
qsub -v EXE_PATH="${exe_path}",\
CSV_FILE="${csv_file}",\
RUNS="${runs}",\
FREQ0="${freq0}",\
FREQ1="${freq1}",\
KERNEL1="${kernel1}",\
KERNEL2="${kernel2}"\
 "${pbs_path}"

# ################ Test 2: Tile 0 -> freq 1600, Tile 1 -> freq 1600, with same workload #############
# freq0=1600
# freq1=1600
# qsub -v EXE_PATH="${exe_path}",\
# CSV_FILE="${csv_file}",\
# ITERS="${iters}",\
# RUNS="${runs}",\
# FREQ0="${freq0}",\
# FREQ1="${freq1}"\
#  "${pbs_path}"

# ################# Test 3: Reduce freq for tile 1 ##############
# freq0=1600
# freq1=800
# qsub -v EXE_PATH="${exe_path}",\
# CSV_FILE="${csv_file}",\
# ITERS="${iters}",\
# RUNS="${runs}",\
# FREQ0="${freq0}",\
# FREQ1="${freq1}"\
#  "${pbs_path}"
