#!/bin/bash

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

script_dir=../rapid-earthquake-dynamics/
job1_id=$(sbatch ${script_dir}/scripts/compact_output_NG.sh | awk '{print $NF}')
echo "Job 1 submitted with ID: $job1_id"
job2_id=$(sbatch --dependency=afterok:$job1_id ${script_dir}/scripts/generate_point_source_representation.sh | awk '{print $NF}')
echo "Job 2 submitted with ID: $job2_id"
job2_id=$(sbatch --dependency=afterok:$job2_id ${script_dir}/scripts/generate_syn.sh | awk '{print $NF}')
echo "Job 3 submitted with ID: $job3_id"
