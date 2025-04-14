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
    with open("derived_config.yaml", "r") as f:
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
    dic_values = {}
    dic_values["mesh_file"] = config_dict["mesh_file"]
    dic_values["mu_delta_min"] = config_dict["mu_delta_min"]
    dic_values["projection"] = config_dict["projection"]

    dic_values["B"] = config_dict["B"]
    dic_values["C"] = config_dict["C"]
    dic_values["R"] = config_dict["R"]
    dic_values["cohesion"] = config_dict["cohesion"]
    mode = config_dict["mode"]

    if "CFS_code" in config_dict:
        CFS_code_fn = config_dict["CFS_code"]
        with open(CFS_code_fn, "r") as f:
            dic_values["CFS_code_placeholder"] = f.read()
    else:
        dic_values["CFS_code_placeholder"] = ""

    generate_input_seissol_dr.generate(mode, dic_values)
