#!/usr/bin/env bash
#
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich
#
# Aggregated SeisSol job launcher for LUMI-G

#SBATCH -J aggregated
#SBATCH --account=project_465002391
#SBATCH --time=00:30:00
#SBATCH -o logs/%j.%x.out
#SBATCH -e logs/%j.%x.out
#SBATCH --partition=standard-g
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mail-type=all
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --export=ALL

set -euo pipefail

##############################
# Environment setup
##############################

if [[ ! -f ./select_gpu ]]; then

  cat <<EOF >select_gpu
#!/bin/bash
export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec "\$@"
EOF
  chmod +x ./select_gpu

fi

CPU_BIND="7e000000000000,7e00000000000000"
CPU_BIND="${CPU_BIND},7e0000,7e000000"
CPU_BIND="${CPU_BIND},7e,7e00"
CPU_BIND="${CPU_BIND},7e00000000,7e0000000000"

export MPICH_GPU_SUPPORT_ENABLED=1
export HSA_XNACK=1
export OMP_NUM_THREADS=3
export OMP_PLACES="cores(3)"
export OMP_PROC_BIND=close
export DEVICE_STACK_MEM_SIZE=4
export SEISSOL_FREE_CPUS_MASK="52-54,60-62,20-22,28-30,4-6,12-14,36-38,44-46"
export PATH=/project/project_465002391/ulrich/seissol_base/seissol/build:$PATH
ulimit -Ss 2097152
unset KMP_AFFINITY

ORDER=${order:-4}
SEISSOL_EXE="SeisSol_Release_sgfx90a_hip_${ORDER}_elastic"

srun_cmd() {
  srun --nodes="$nodes_per_job" \
    --nodelist="$node_subset" \
    --ntasks="$tasks_per_job" \
    --ntasks-per-node=8 \
    --gpus-per-node=8 \
    --cpus-per-task=7 \
    --cpu-bind=mask_cpu:${CPU_BIND} \
    -o "./logs/${SLURM_JOB_ID}_runs/$logfile" \
    --exclusive \
    ./select_gpu "$SEISSOL_EXE" "$filename"
}

script_dir=../rapid-earthquake-dynamics/
source $script_dir/scripts/common/aggregated_jobs.sh "$@"
