#!/bin/bash

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

# Job Name and Files (also --job-name)
#SBATCH -J compact
#Output and error (also --output, --error):
#SBATCH -o ./%j.%x.out
#SBATCH -e ./%j.%x.out

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
    counter=$((counter+1))
    echo "Processing file $counter of $total_params: $current_file"
    srun -N 1 -n 1 -c 1 --exclusive --mem-per-cpu 8G seissol_output_extractor $current_file --time "i:" --variable u1 u2 u3 --add2prefix _disp &
    # Improved check: avoids unnecessary wait on the first iteration
    if (( $counter >= $SLURM_NTASKS )); then
      echo "waiting, $counter"
      wait -n
    fi
done


for current_file in output/*-fault.xdmf; do
    counter=$((counter+1))
    echo "Processing file $counter of $total_params: $current_file"
    srun -N 1 -n 1 -c 1 --exclusive --mem-per-cpu 8G seissol_output_extractor $current_file &
    if (( $counter >= $SLURM_NTASKS )); then
      echo "waiting, $counter"
      wait -n
    fi
done

for current_file in output/dyn_*-energy.csv; do
    counter=$((counter+1))
    echo "Processing file $counter of $total_params: $current_file"
    srun -N 1 -n 1 -c 1  --exclusive --mem-per-cpu 8G cp $current_file extracted_output &
    if (( $counter >= $SLURM_NTASKS )); then
      echo "waiting, $counter"
      wait -n
    fi
done
wait

mv *_extracted* extracted_output 
mv *_disp* extracted_output
#mv output/*-receiver-* extracted_output
find . -maxdepth 1 -name "*output/*-receiver-*" -exec mv {} extracted_output \; || echo "No files to move."
wait

echo "now generate point source representation"

proj=$(grep '^projection:' derived_config.yaml | cut -d ':' -f2 | xargs)
script_dir=../rapid-earthquake-dynamics/
echo $script_dir   

# Create an indexed list of files
files=(extracted_output/dyn*-fault.xdmf)
num_files=${#files[@]}
echo "Found $num_files files to process."

# Process files in parallel
for filename in "${files[@]}"; do
    echo "Processing file: $filename"
    
    # Launch task in the background
    srun --nodes=1 -n 1 -c 1 --exclusive --mem-per-cpu 8G \
        $script_dir/submodules/seismic-waveform-factory/scripts/compute_multi_cmt.py \
        spatial "$filename" yaml_files/material.yaml --DH 10 --proj "${proj}" --NZ 4 &

    # Increment counter
    counter=$((counter + 1))
    
    if (( $counter >= $SLURM_NTASKS )); then
    # Wait after every SLURM_NTASKS tasks
        echo "waiting, $counter"
        wait -n
    fi
done



# Wait for remaining background jobs
echo "Waiting for remaining tasks..."
wait

# Collect output after all tasks complete
mkdir -p mps_regional
mv PointSource* mps_regional


# Process files in parallel
for filename in "${files[@]}"; do
    echo "Processing file: $filename"
    
    # Launch task in the background
    srun --nodes=1 -n 1 -c 1 --exclusive --mem-per-cpu 8G \
        $script_dir/submodules/seismic-waveform-factory/scripts/compute_multi_cmt.py \
        spatial "$filename" yaml_files/material.yaml --DH 10 --proj "${proj}" --NZ 4 \
        --slip_threshold " -1e10" --use_geometric_center &

    # Increment counter
    counter=$((counter + 1))
    
    if (( $counter >= $SLURM_NTASKS )); then
    # Wait after every SLURM_NTASKS tasks
        echo "waiting, $counter"
        wait -n
    fi
done

# Wait for remaining background jobs
echo "Waiting for remaining tasks..."
wait

# Collect output after all tasks complete
mkdir -p mps_teleseismic
mv PointSource* mps_teleseismic

echo "All tasks completed!"
