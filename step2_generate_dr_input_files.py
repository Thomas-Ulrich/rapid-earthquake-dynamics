#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import os
import sys

import yaml

from dynworkflow import generate_input_seissol_dr

# Append kinematic_models folder to path
# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "dynworkflow/kinematic_models"
absolute_path = os.path.join(current_script_dir, relative_path)
if absolute_path not in sys.path:
    sys.path.append(absolute_path)

from kinematic_models import project_fault_tractions_onto_asagi_grid

if __name__ == "__main__":
    with open("derived_config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    fault_mesh_size = config_dict["fault_mesh_size"]

    fl33_file_candidates = [
        "output_fl33/fl33-fault.xdmf",
        "output/fl33-fault.xdmf",
        "extracted_output/fl33_compacted-fault.xdmf",
        "extracted_output/fl33_extracted-fault.xdmf",
    ]

    try:
        fl33_file = next(f for f in fl33_file_candidates if os.path.exists(f))
    except StopIteration:
        raise FileNotFoundError(
            (
                "None of the fl33-fault.xdmf files were found "
                "in the expected directories."
            )
        )

    project_fault_tractions_onto_asagi_grid.generate_input_files(
        fl33_file,
        fault_mesh_size / 2,
        7,
        gaussian_kernel=fault_mesh_size,
        taper=None,
        paraview_readable=None,
        edge_clearance=8,
    )

    generate_input_seissol_dr.generate()
