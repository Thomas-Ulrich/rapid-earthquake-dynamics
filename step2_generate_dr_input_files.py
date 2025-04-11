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
    with open("derived_config", "r") as f:
        config_dict = yaml.safe_load(f)
    fault_mesh_size = config_dict["fault_mesh_size"]

    fl33_file = "output/fl33-fault.xdmf"
    if not os.path.exists(fl33_file):
        fl33_file = "extracted_output/fl33_extracted-fault.xdmf"

    project_fault_tractions_onto_asagi_grid.generate_input_files(
        fl33_file,
        fault_mesh_size / 2,
        gaussian_kernel=fault_mesh_size,
        taper=None,
        paraview_readable=None,
    )
    mode = "grid_search"
    # mode = 'latin_hypercube'
    # mode = "picked_models"
    dic_values = {}
    mesh_file = dic_values["mesh_file"]
    dic_values["mu_delta_min"] = config_dict["mu_delta_min"]
    if "CFS_code" in config_dict:
        CFS_code_fn = config_dict["CFS_code"]
        with open("your_file.txt", "r") as f:
            dic_values["CFS_code_placeholder"] = f.read()
    else:
        dic_values["CFS_code_placeholder"] = ""

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

        generate_input_seissol_dr.generate(mode, dic_values)
