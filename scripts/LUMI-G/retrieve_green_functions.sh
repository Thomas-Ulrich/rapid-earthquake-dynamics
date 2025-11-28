#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

#SBATCH -J retrieve_green
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

echo 'num_nodes:' $SLURM_JOB_NUM_NODES 'ntasks:' $SLURM_NTASKS
ulimit -Ss 2097152

proj=$(grep '^projection:' derived_config.yaml | cut -d ':' -f2 | xargs)

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

filename=output/dyn-kinmod-fault.xdmf

swf compute-multi-cmt spatial "$filename" yaml_files/material.yaml \
  --DH 20 --proj "${proj}" --NZ 4 --slip_threshold " -1e10" \
  --use_geometric_center $refVectorArgs

# Collect output after all tasks complete
mkdir -p mps_teleseismic
mv PointSource* mps_teleseismic

redyn add-sources
if [ -f waveforms_config_teleseismic.yaml ]; then
  swf plot-waveforms waveforms_config_teleseismic.yaml
fi
