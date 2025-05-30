#!/bin/bash

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

# Job Name and Files (also --job-name)
#SBATCH -J 2015_1.1
#Output and error (also --output, --error):
#SBATCH -o ./%j.%x.out
#SBATCH -e ./%j.%x.out

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
#SBATCH --nodes=40 --partition=general --time=01:00:00
##SBATCH --nodes=16 --partition=test --time=00:30:00 --exclude="i01r01c01s[01-12]"

module load slurm_setup

#Run the program:
export MP_SINGLE_THREAD=no
unset KMP_AFFINITY
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

echo 'num_nodes:' $SLURM_JOB_NUM_NODES 'ntasks:' $SLURM_NTASKS
ulimit -Ss 2097152

ORDER=${order:-4}
module load seissol/1.3.1-intel23-o${ORDER}-elas-dunav-single-impi

part_file=$1

mapfile -t filenames < "$part_file"

# Iterate over the array of filenames
for filename in "${filenames[@]}"; do
    echo "Processing file: $filename"
    srun SeisSol_Release_sskx_${ORDER}_elastic $filename

    # Extract the core part of the filename by removing 'parameters_' and '.par'
    #core_name=$(basename "$filename" .par)
    #core_name=${core_name#parameters_}

    # Construct the expected output file path
    #output_file="output/${core_name}-energy.csv"

    # Check if the output file exists
    # If the output file does not exist, process the file
    #if [ ! -f "$output_file" ]; then
    #    echo "something went wrong? trying rerun seissol with file: $filename"
    #    srun SeisSol_Release_sskx_5_elastic $filename
    #fi
done
