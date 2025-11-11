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
##SBATCH --nodes=16 --partition=test --time=00:30:00

module load slurm_setup
# use a number of nodes multiple of 16!
if ((SLURM_JOB_NUM_NODES % 16 != 0)); then
    echo "$SLURM_JOB_NUM_NODES not a multiple of 16"
    exit 1
fi

if [ "$SLURM_JOB_NUM_NODES" -lt 208 ]; then
    ndivide=$((SLURM_JOB_NUM_NODES / 8))
else
    ndivide=$((SLURM_JOB_NUM_NODES / 16))
fi

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

echo 'num_nodes:' $SLURM_JOB_NUM_NODES 'ntasks:' $SLURM_NTASKS
ulimit -Ss 2097152

ORDER=${order:-4}
module load seissol/1.3.1-oneapi25-o${ORDER}-elas-dunav-single-impi
#module load seissol/master-oneapi25-o${ORDER}-elas-dunav-single-impi
unset KMP_AFFINITY

nodes_per_job=$(($SLURM_JOB_NUM_NODES / $ndivide))
tasks_per_job=$(($nodes_per_job * 2))

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

    srun -B 2:48:2 -c 48 --nodes=$nodes_per_job --ntasks=$tasks_per_job \
        --ntasks-per-node=2 --exclusive \
        -o ./logs/$SLURM_JOB_ID.$counter0.$id.out \
        SeisSol_Release_sskx_${ORDER}_elastic "$filename"
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
