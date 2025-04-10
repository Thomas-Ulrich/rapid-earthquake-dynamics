#!/usr/bin/env python3
from dynworkflow import generate_input_seissol_dr
import sys
import os
import yaml

# Append kinematic_models folder to path
# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "dynworkflow/kinematic_models"
absolute_path = os.path.join(current_script_dir, relative_path)
if absolute_path not in sys.path:
    sys.path.append(absolute_path)

import project_fault_tractions_onto_asagi_grid

if __name__ == "__main__":
    with open("tmp/inferred_fault_mesh_size.txt", "r") as f:
        inferred_fault_mesh_size = float(f.read())

    fl33_file = "output/fl33-fault.xdmf"
    if not os.path.exists(fl33_file):
        fl33_file = "extracted_output/fl33_extracted-fault.xdmf"

    project_fault_tractions_onto_asagi_grid.generate_input_files(
        fl33_file,
        inferred_fault_mesh_size / 2,
        gaussian_kernel=inferred_fault_mesh_size,
        taper=None,
        paraview_readable=None,
    )
    mode = "grid_search"
    # mode = 'latin_hypercube'
    # mode = "picked_models"
    dic_values = {}
    if mode == "picked_models":
        dic_values["B"] = [0.9, 1.0, 1.2]
        dic_values["C"] = [0.3 for i in range(3)]
        dic_values["R"] = [0.65 for i in range(3)]
        dic_values["cohesion"] = [(0.25, 0), (0.25, 1), (0.25, 2.5)]
        generate_input_seissol_dr.generate(mode, dic_values)
    else:
        dic_values["B"] = [0.9, 1.0, 1.1, 1.2]
        dic_values["C"] = [0.1, 0.2, 0.3, 0.4, 0.5]
        dic_values["R"] = [0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
        # dic_values["cohesion"] = [(0.25, 0)]
        dic_values["cohesion"] = [(0.25, 1)]
        # only relevant for Latin Hypercube
        dic_values["nsamples"] = 50
        with open("config.yaml", "r") as f:
            config_dict = yaml.safe_load(f)
        dic_values["mu_delta_min"] = config_dict["mu_delta_min"]

        generate_input_seissol_dr.generate(mode, dic_values)
