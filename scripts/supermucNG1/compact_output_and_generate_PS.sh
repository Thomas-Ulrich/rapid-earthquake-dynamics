#!/bin/bash

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

# Job Name and Files (also --job-name)
#SBATCH -J compact
#Output and error (also --output, --error):
#SBATCH -o logs/%j.%x.out
#SBATCH -e logs/%j.%x.out

#Initial working directory:
#SBATCH --chdir=./

#Notification and type
#SBATCH --mail-type=END
# Wall clock limit:
#SBATCH --time=00:30:00
#SBATCH --no-requeue

#Setup of execution environment
#SBATCH --export=ALL
#SBATCH --account=pn49ha
#SBATCH --partition=test

#Number of nodes and MPI tasks per node:
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=10
#EAR may impact code performance
#SBATCH --ear=off
#SBATCH --mem=80G
tasks_per_node=10 # Change this to how many parallel jobs you want per node

set -euo pipefail
mkdir -p extracted_output

counter=0
total_params=$(ls output/dyn_*-surface.xdmf | wc -l)
total_params=$((total_params * 3))

# First sanity check to find if any nan at t=0
pattern="^0,plastic_moment,0"
for current_file in output/dyn_*-energy.csv; do
    if tail -n 1 "$current_file" | grep -q "$pattern"; then
        base_filename="${current_file%-energy.csv}"
        echo "no output detected in $current_file, removing... $base_filename*"
        rm $base_filename*
    fi
done

for current_file in output/dyn_*-surface.xdmf; do
    counter=$((counter + 1))
    echo "Processing file $counter of $total_params: $current_file"
    srun -N 1 -n 1 -c 1 --exclusive --mem-per-cpu 8G seissol_output_extractor $current_file --time "i:" --variable u1 u2 u3 --add2prefix _disp &
    # Improved check: avoids unnecessary wait on the first iteration
    if (($counter >= $SLURM_NTASKS)); then
        echo "waiting, $counter"
        wait -n
    fi
done

for current_file in output/*-fault.xdmf; do
    counter=$((counter + 1))
    echo "Processing file $counter of $total_params: $current_file"
    srun -N 1 -n 1 -c 1 --exclusive --mem-per-cpu 8G seissol_output_extractor $current_file --add2prefix _compacted &
    if (($counter >= $SLURM_NTASKS)); then
        echo "waiting, $counter"
        wait -n
    fi
done

for current_file in output/dyn_*-energy.csv; do
    counter=$((counter + 1))
    echo "Processing file $counter of $total_params: $current_file"
    srun -N 1 -n 1 -c 1 --exclusive --mem-per-cpu 8G cp $current_file extracted_output &
    if (($counter >= $SLURM_NTASKS)); then
        echo "waiting, $counter"
        wait -n
    fi
done
wait

mv *_compacted* extracted_output
mv *_disp* extracted_output
find . -maxdepth 1 -name "*output/*receiver-*" -exec mv {} extracted_output \; || echo "No files to move."
wait

echo "generating point source representation"

proj=$(grep '^projection:' derived_config.yaml | cut -d ':' -f2 | xargs)
script_dir=../rapid-earthquake-dynamics/
echo $script_dir

# Create an indexed list of files
files=(extracted_output/dyn*-fault.xdmf)
num_files=${#files[@]}
echo "Found $num_files files to process."

# Configuration

# Get node list
nodes=($(scontrol show hostname "$SLURM_JOB_NODELIST"))
num_nodes=${#nodes[@]}

echo "Nodes allocated: ${nodes[*]}"
echo "Number of files: ${#files[@]}"

job_idx=0

# Extract values using grep + awk
XRef=$(grep -E '^XRef' parameters_fl33.par | awk -F= '{print $2}' | xargs)
YRef=$(grep -E '^YRef' parameters_fl33.par | awk -F= '{print $2}' | xargs)
ZRef=$(grep -E '^ZRef' parameters_fl33.par | awk -F= '{print $2}' | xargs)
refPointMethod=$(grep -E '^refPointMethod' parameters_fl33.par | awk -F= '{print $2}' | xargs)

# Check if refPointMethod == 1
if [ "$refPointMethod" -eq 1 ]; then
    echo "Using reference vector: $XRef $YRef $ZRef"
    refVectorArgs="--refVector $XRef $YRef $ZRef"
else
    echo "refPointMethod != 1, skipping refVector"
    refVectorArgs=""
fi

for filename in "${files[@]}"; do
    # Determine node and local index
    node_idx=$((job_idx / tasks_per_node % num_nodes))
    node=${nodes[$node_idx]}

    echo "[$job_idx] Launching on $node: $filename"

    srun --nodes=1 --nodelist="$node" -n 1 -c 1 --exclusive --mem-per-cpu=8G \
        swf compute-multi-cmt spatial "$filename" yaml_files/material.yaml \
        --DH 20 --proj "${proj}" --NZ 4 $refVectorArgs &

    job_idx=$((job_idx + 1))
    # Wait when too many jobs are running in parallel
    if ((job_idx % (num_nodes * tasks_per_node) == 0)); then
        echo "Waiting for a batch to complete..."
        wait
    fi
done

# Wait for remaining background jobs
echo "Waiting for remaining tasks..."
wait

# Collect output after all tasks complete
mkdir -p mps_regional
mv PointSource* mps_regional

job_idx=0

for filename in "${files[@]}"; do
    # Determine node and local index
    node_idx=$((job_idx / tasks_per_node % num_nodes))
    node=${nodes[$node_idx]}

    echo "[$job_idx] Launching on $node: $filename"

    srun --nodes=1 --nodelist="$node" -n 1 -c 1 --exclusive --mem-per-cpu=8G \
        swf compute-multi-cmt spatial "$filename" yaml_files/material.yaml \
        --DH 20 --proj "${proj}" --NZ 4 --slip_threshold " -1e10" \
        --use_geometric_center $refVectorArgs &

    job_idx=$((job_idx + 1))
    # Wait when too many jobs are running in parallel
    if ((job_idx % (num_nodes * tasks_per_node) == 0)); then
        echo "Waiting for a batch to complete..."
        wait
    fi
done

# Wait for remaining background jobs
echo "Waiting for remaining tasks..."
wait

# Collect output after all tasks complete
mkdir -p mps_teleseismic
mv PointSource* mps_teleseismic

echo "All tasks completed!"
