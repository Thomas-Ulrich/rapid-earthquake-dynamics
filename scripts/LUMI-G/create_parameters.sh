#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

#SBATCH -J gen_dr_input
#SBATCH --nodes=1               # Total number of nodes
#SBATCH --account=project_465002391  # Project for billing
#SBATCH --time=00:30:00       # Run time (d-hh:mm:ss)
#SBATCH -o logs/%j.%x.out
#SBATCH -e logs/%j.%x.out
#SBATCH --partition=small-g  # Partition (queue) name
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node
#SBATCH --gpus-per-node=0       # Allocate one gpu per MPI rank
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --export=ALL

#Run the program:
unset KMP_AFFINITY
export MP_SINGLE_THREAD=no
export OMP_NUM_THREADS=1

echo 'num_nodes:' $SLURM_JOB_NUM_NODES 'ntasks:' $SLURM_NTASKS
ulimit -Ss 2097152

srun -u python ../rapid-earthquake-dynamics/src/dynworkflow/generate_input_files_for_dr_ensemble.py
