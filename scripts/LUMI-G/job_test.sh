#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

#SBATCH --job-name=fl33   # Job name
#SBATCH --nodes=1               # Total number of nodes
#SBATCH --account=project_465002391  # Project for billing
#SBATCH --time=01:00:00       # Run time (d-hh:mm:ss)
#SBATCH -o logs/%j.%x.out
#SBATCH -e logs/%j.%x.out
#SBATCH --partition=small-g  # Partition (queue) name
#SBATCH --ntasks-per-node=8     # 8 MPI ranks per node
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --export=ALL

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
export PATH=$SEISSOL_BASE/seissol/build_optim:$PATH
ulimit -Ss 2097152

part_file=$1
ORDER=${order:-5}

mapfile -t filenames <"$part_file"

# Iterate over the array of filenames
for filename in "${filenames[@]}"; do
  echo "Processing file: $filename"
  srun --cpu-bind=mask_cpu:${CPU_BIND} seissol-launch SeisSol_Release_sgfx90a_hip_${ORDER}_elastic $filename
  # Extract the core part of the filename by removing 'parameters_' and '.par'
  core_name=$(basename "$filename" .par)
  core_name=${core_name#parameters_}

  # Construct the expected output file path
  output_file="output/${core_name}-energy.csv"

  # Check if the output file exists
  # If the output file does not exist, process the file
  #if [ ! -f "$output_file" ]; then
  #    echo "something went wrong? trying rerun seissol with file: $filename"
  #    srun SeisSol_Release_sskx_4_elastic $filename
  #fi
done
