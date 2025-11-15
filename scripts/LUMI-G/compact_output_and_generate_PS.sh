#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

#SBATCH -J compact
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --account=project_465002391  # Project for billing
#SBATCH --time=00:30:00       # Run time (d-hh:mm:ss)
#SBATCH -o logs/%j.%x.out
#SBATCH -e logs/%j.%x.out
#SBATCH --partition=small-g  # Partition (queue) name
#SBATCH --ntasks-per-node=56     # 8 MPI ranks per node
#SBATCH --gpus-per-node=0       # Allocate one gpu per MPI rank
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --mem=480G

#Run the program:
unset KMP_AFFINITY
export MP_SINGLE_THREAD=no
export OMP_NUM_THREADS=1

script_dir=../rapid-earthquake-dynamics/
bash $script_dir/scripts/common/compact_output_and_generate_PS.sh $SLURM_NTASKS_PER_NODE
