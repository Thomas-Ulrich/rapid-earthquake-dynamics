#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

#SBATCH -J gen_syn
#SBATCH --nodes=1               # Total number of nodes
#SBATCH --account=project_465002391  # Project for billing
#SBATCH --time=00:30:00       # Run time (d-hh:mm:ss)
#SBATCH -o logs/%j.%x.out
#SBATCH -e logs/%j.%x.out
#SBATCH --partition=small-g  # Partition (queue) name
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --mem=480G

export OMP_NUM_THREADS=64

#bash $(dirname "$0")/../common/generate_syn.sh
script_dir=../rapid-earthquake-dynamics/
bash $script_dir/scripts/common/generate_syn.sh
