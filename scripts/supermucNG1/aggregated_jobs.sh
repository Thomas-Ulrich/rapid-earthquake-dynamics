#!/bin/bash

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

# Job Name and Files (also --job-name)
#SBATCH -J aggregated
#Output and error (also --output, --error):
#SBATCH -o ./logs/%j.%x.out
#SBATCH -e ./logs/%j.%x.out

#Initial working directory:
#SBATCH --chdir=./

#Notification and type
#SBATCH --mail-type=END
#SBATCH --mail-user=thomas.ulrich@lmu.de
#SBATCH --no-requeue
#SBATCH --export=ALL
#SBATCH --account=pn49ha
#SBATCH --ntasks-per-node=2
#EAR may impact code performance
#SBATCH --ear=off
#SBATCH --partition=general

module load slurm_setup

#Run the program:
export MP_SINGLE_THREAD=no
export OMP_NUM_THREADS=46
export OMP_PLACES="cores(23)"
#Prevents errors such as experience in Issue #691
export I_MPI_SHM_HEAP_VSIZE=8192

export XDMFWRITER_ALIGNMENT=8388608
export XDMFWRITER_BLOCK_SIZE=8388608
export SC_CHECKPOINT_ALIGNMENT=8388608

export SEISSOL_CHECKPOINT_ALIGNMENT=8388608
export SEISSOL_CHECKPOINT_DIRECT=1
export ASYNC_MODE=THREAD
export ASYNC_BUFFER_ALIGNMENT=8388608
source /etc/profile.d/modules.sh

ulimit -Ss 2097152

module load seissol/1.3.1-oneapi25-o${ORDER}-elas-dunav-single-impi
#module load seissol/master-oneapi25-o${ORDER}-elas-dunav-single-impi
unset KMP_AFFINITY

# different SeisSol executable
SEISSOL_EXE="SeisSol_Release_sskx_${ORDER}_elastic"

srun_cmd() {
  srun -B 2:48:2 -c 48 \
    --nodes="$nodes_per_job" \
    --nodelist="$node_subset" \
    --ntasks="$tasks_per_job" \
    --ntasks-per-node=2 \
    -o "./logs/${SLURM_JOB_ID}_runs/$logfile" \
    --exclusive \
    "$SEISSOL_EXE" "$filename"
}

script_dir=../rapid-earthquake-dynamics/
bash $script_dir/scripts/common/aggregated_jobs.sh "$@"
