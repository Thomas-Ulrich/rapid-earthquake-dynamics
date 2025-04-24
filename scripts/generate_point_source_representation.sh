#!/bin/bash

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

# Job Name and Files
#SBATCH -J PSrep
#SBATCH -o ./%j.%x.out
#SBATCH -e ./%j.%x.out
#SBATCH --chdir=./
#SBATCH --mail-type=END
#SBATCH --mail-user=thomas.ulrich@lmu.de
#SBATCH --no-requeue
#SBATCH --export=ALL
#SBATCH --account=pn49ha
#SBATCH --ntasks-per-node=10
#SBATCH --ear=off
#SBATCH --nodes=4 --partition=test --time=00:30:00 --exclude="i01r01c01s[01-07]"
#SBATCH --mem=80G

source /etc/profile.d/modules.sh
module load slurm_setup

echo 'num_nodes:' $SLURM_JOB_NUM_NODES 'ntasks:' $SLURM_NTASKS
ulimit -Ss 2097152

proj=$(cat "tmp/projection.txt")
script_dir=../rapid-earthquake-dynamics/
echo $script_dir   

# Create an indexed list of files
files=(extracted_output/dyn*-fault.xdmf)
num_files=${#files[@]}
echo "Found $num_files files to process."

# Process files in parallel
counter=0
for filename in "${files[@]}"; do
    echo "Processing file: $filename"
    
    # Launch task in the background
    srun --nodes=1 -n 1 -c 1 --exclusive --mem-per-cpu 8G \
        $script_dir/submodules/seismic-waveform-factory/scripts/compute_multi_cmt.py \
        spatial "$filename" 1 tmp/depth_vs_rigidity.txt --DH 10 --proj "${proj}" --NZ 4 &
    
    # Increment counter
    counter=$((counter + 1))
    
    # Wait after every SLURM_NTASKS tasks
    if (( $counter % $SLURM_NTASKS == 0 )); then
        echo "Waiting for batch of $SLURM_NTASKS tasks to finish..."
        wait
    fi
done

# Wait for remaining background jobs
echo "Waiting for remaining tasks..."
wait

# Collect output after all tasks complete
mv PointSource* tmp
echo "All tasks completed!"

