#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

# Common workflow for generating synthetic data

set -euo pipefail

ulimit -Ss 2097152

export MP_SINGLE_THREAD=no
unset KMP_AFFINITY

output_dir=extracted_output
script_dir=../rapid-earthquake-dynamics/

if [ -f $output_dir/dyn-kinmod_compacted-fault.xdmf ]; then
  $script_dir/src/dynworkflow/compute_gof_fault_slip.py $output_dir/dyn_ $output_dir/dyn-kinmod_compacted-fault.xdmf
else
  $script_dir/src/dynworkflow/compute_gof_fault_slip.py $output_dir/dyn_ $output_dir/dyn-kinmod_extracted-fault.xdmf
fi

$script_dir/src/dynworkflow/compute_percentage_supershear.py $output_dir/dyn_ yaml_files/material.yaml

if [ -f offsets.csv ]; then
  $script_dir/src/dynworkflow/compare_offset.py $output_dir/dyn_ offsets.csv
fi

$script_dir/src/dynworkflow/add_source_files_to_waveform_config.py

export OMP_NUM_THREADS=$(grep -c ^processor /proc/cpuinfo)

if [ -f waveforms_config_regional.yaml ]; then
  swf plot-waveforms waveforms_config_regional.yaml
fi

if [ -f waveforms_config_teleseismic.yaml ]; then
  swf plot-waveforms waveforms_config_teleseismic.yaml
fi

$script_dir/src/dynworkflow/compile_scenario_macro_properties.py $output_dir
