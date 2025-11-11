#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

#SBATCH -J aggregated
#SBATCH --account=project_465002391  # Project for billing
#SBATCH --time=00:30:00       # Run time (d-hh:mm:ss)
#SBATCH -o logs/%j.%x.out
#SBATCH -e logs/%j.%x.out
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --ntasks-per-node=8     # 8 MPI ranks per node
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --export=ALL

cat <<EOF >select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF

chmod +x ./select_gpu

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

# use a number of nodes multiple of 16!
if ((SLURM_JOB_NUM_NODES % 2 != 0)); then
    echo "$SLURM_JOB_NUM_NODES not a multiple of 2"
    exit 1
fi

if [ "$SLURM_JOB_NUM_NODES" -lt 60 ]; then
    ndivide=$((SLURM_JOB_NUM_NODES / 1))
else
    ndivide=$((SLURM_JOB_NUM_NODES / 2))
fi

ulimit -Ss 2097152

ORDER=${order:-4}

unset KMP_AFFINITY

nodes_per_job=$(($SLURM_JOB_NUM_NODES / $ndivide))
tasks_per_job=$(($nodes_per_job * 8))

if [[ -n "$1" ]]; then
    part_file=$1
    echo "$(date '+%Y-%m-%d %H:%M:%S') - reading parameter files from: $1"
    mapfile -t files <"$part_file"
else
    files=(parameters_dyn_*.par)
fi

num_files=${#files[@]}
echo "Found $num_files files to process."

run_file() {
    local filename=$1
    local counter=$2
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Processing file: $filename"
    local counter0=$(printf "%05d" "$counter")
    local id=$(echo "$filename" | sed -n 's/^parameters_dyn_\([0-9]\{4\}\)_.*\.par/\1/p')

    srun --nodes=$nodes_per_job --ntasks=$tasks_per_job \
        --ntasks-per-node=8 --gpus-per-node=8 \
        -o ./logs/$SLURM_JOB_ID.$counter0.$id.out --exclusive --cpu-bind=mask_cpu:${CPU_BIND} ./select_gpu SeisSol_Release_sgfx90a_hip_${ORDER}_elastic $filename
}

# Process files in parallel
counter=0
for filename in "${files[@]}"; do
    run_file "$filename" "$counter" & # run in background
    counter=$((counter + 1))

    # Ensure we don’t exceed max concurrent jobs
    if (($counter >= $ndivide)); then
        wait -n # Wait for the first finished job before launching a new one
    fi
done

wait
counter=0
# Iterate over the array of filenames
for filename in "${files[@]}"; do
    # Extract the core part of the filename by removing 'parameters_' and '.par'
    core_name=$(basename "$filename" .par)
    core_name=${core_name#parameters_}

    # Construct the expected output file path
    output_file="output/${core_name}-energy.csv"

    # Check if the output file exists
    # If the output file does not exist, process the file
    if [ ! -f "$output_file" ]; then
        echo "something went wrong? trying rerun seissol with file: $filename"
        run_file "$filename" "$counter" & # run in background
        counter=$((counter + 1))
    fi

    # Ensure we don’t exceed max concurrent jobs
    if (($counter >= $ndivide)); then
        wait -n # Wait for the first finished job before launching a new one
    fi
done
