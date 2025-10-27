#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import os

import numpy as np

# Step 1: Collect all parameter files
parameter_files = sorted([f for f in os.listdir(".") if f.startswith("parameters_dyn")])
nfiles = len(parameter_files)
# Step 2: Define the number of parts
n = 3  # Adjust this to the number of parts you want to split into

# Step 3: Split the list of files into n parts with contiguous indexes
parts = np.array_split(np.arange(nfiles), n)
split_files = [list(np.array(parameter_files)[part]) for part in parts]


current_script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "scripts/job_NG.sh"
absolute_path_2_job = os.path.join(current_script_dir, relative_path)

# Step 4: Write each part to a separate file and process it
for i, part in enumerate(split_files):
    part_filename = f"part_{i + 1}.txt"
    with open(part_filename, "w") as f:
        for par_file in part:
            f.write(par_file + "\n")
    command = f"sbatch {absolute_path_2_job} {part_filename}"
    print(command)
    os.system(command)
