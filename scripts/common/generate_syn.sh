#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

# Common workflow for generating synthetic data

set -euo pipefail

ulimit -Ss 2097152

export MP_SINGLE_THREAD=no
unset KMP_AFFINITY

output_dir=extracted_output

if [ -f $output_dir/dyn-kinmod_compacted-fault.xdmf ]; then
  ref_model=$output_dir/dyn-kinmod_compacted-fault.xdmf
else
  ref_model=$output_dir/dyn-kinmod_extracted-fault.xdmf
fi
redyn metrics slip $output_dir/dyn_ $ref_model

redyn metrics supershear $output_dir/dyn_ yaml_files/material.yaml

if [ -f offsets.csv ]; then
  redyn metrics fault-offsets $output_dir/dyn_ offsets.csv
fi

redyn add-sources

export OMP_NUM_THREADS=$(grep -c ^processor /proc/cpuinfo)

if [ -f waveforms_config_regional.yaml ]; then
  swf plot-waveforms waveforms_config_regional.yaml
fi

if [ -f waveforms_config_teleseismic.yaml ]; then
  swf plot-waveforms waveforms_config_teleseismic.yaml
fi

redyn metrics rank_models $output_dir
