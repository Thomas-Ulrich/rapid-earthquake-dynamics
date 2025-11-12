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

echo 'num_nodes:' $SLURM_JOB_NUM_NODES 'ntasks:' $SLURM_NTASKS
ulimit -Ss 2097152

export MP_SINGLE_THREAD=no
unset KMP_AFFINITY
export OMP_NUM_THREADS=48

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
if [ -f waveforms_config_regional.yaml ]; then
    swf plot-waveforms waveforms_config_regional.yaml
fi
if [ -f waveforms_config_teleseismic.yaml ]; then
    swf plot-waveforms waveforms_config_teleseismic.yaml
fi
$script_dir/dynworkflow/compile_scenario_macro_properties.py $output_dir
