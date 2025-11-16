#!/bin/bash

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

# Job Name and Files (also --job-name)
#SBATCH -J gen_syn
#Output and error (also --output, --error):
#SBATCH -o logs/%j.%x.out
#SBATCH -e logs/%j.%x.out

#Initial working directory:
#SBATCH --chdir=./

#Notification and type
#SBATCH --mail-type=END
#SBATCH --mail-user=thomas.ulrich@lmu.de
#SBATCH --no-requeue
#SBATCH --export=ALL
#SBATCH --account=pn49ha
#SBATCH --ntasks-per-node=1
#EAR may impact code performance
#SBATCH --ear=off
##SBATCH --nodes=20 --partition=general --time=02:30:00
#SBATCH --nodes=1 --partition=test --time=00:30:00  --exclude="i01r01c01s[01-08]"

#SBATCH --mem=80G

source /etc/profile.d/modules.sh

module load slurm_setup

export OMP_NUM_THREADS=48
script_dir=../rapid-earthquake-dynamics/
bash $script_dir/scripts/common/generate_syn.sh
