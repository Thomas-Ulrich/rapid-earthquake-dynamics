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
#SBATCH --mem=80G


echo 'num_nodes:' $SLURM_JOB_NUM_NODES 'ntasks:' $SLURM_NTASKS
ulimit -Ss 2097152

export MP_SINGLE_THREAD=no
unset KMP_AFFINITY
export OMP_NUM_THREADS=64

output_dir=extracted_output
script_dir=../rapid-earthquake-dynamics/
if [ -f $output_dir/dyn-kinmod_compacted-fault.xdmf ]; then
    $script_dir/dynworkflow/compute_gof_fault_slip.py $output_dir/dyn_ $output_dir/dyn-kinmod_compacted-fault.xdmf
else
    #backwards compatibility
    $script_dir/dynworkflow/compute_gof_fault_slip.py $output_dir/dyn_ $output_dir/dyn-kinmod_extracted-fault.xdmf
fi

$script_dir/dynworkflow/compute_percentage_supershear.py $output_dir/dyn_ yaml_files/material.yaml
if [ -f offsets.csv ]; then
  $script_dir/dynworkflow/compare_offset.py $output_dir/dyn_ offsets.csv
fi

$script_dir/dynworkflow/add_source_files_to_waveform_config.py
export OMP_NUM_THREADS=$(grep -c ^processor /proc/cpuinfo)
if [ -f waveforms_config_regional_sources.yaml ]; then
    swf plot-waveforms waveforms_config_regional_sources.yaml
fi
if [ -f waveforms_config_teleseismic_sources.yaml ]; then
    swf plot-waveforms waveforms_config_teleseismic_sources.yaml
fi
$script_dir/dynworkflow/compile_scenario_macro_properties.py $output_dir
