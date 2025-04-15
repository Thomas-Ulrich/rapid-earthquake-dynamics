#!/usr/bin/env python3
from dynworkflow import generate_input_seissol_dr
import sys
import os
import yaml
import re

# Append kinematic_models folder to path
# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "dynworkflow/kinematic_models"
absolute_path = os.path.join(current_script_dir, relative_path)
if absolute_path not in sys.path:
    sys.path.append(absolute_path)

import project_fault_tractions_onto_asagi_grid


def parse_parameter_string(param_str):
    dic_values = {}
    for match in re.finditer(r"(\w+)=([^\s]+)", param_str):
        key, val = match.group(1), match.group(2)
        if key == "cohesion":
            dic_values["cohesion"] = [
                list(map(float, pair.split(","))) for pair in val.split(";")
            ]
        else:
            dic_values[key] = [float(v) for v in val.split(",") if v.strip()]
    print(dic_values)
    return dic_values


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
    dic_values["projection"] = config_dict["projection"]

    with open("input_config.yaml", "r") as f:
        input_config_dict = yaml.safe_load(f)
    dic_values |= parse_parameter_string(input_config_dict["parameters"])
    mode = input_config_dict["mode"]
    dic_values["mu_delta_min"] = input_config_dict["mu_delta_min"]
    dic_values["mu_d"] = input_config_dict["mu_d"]

    if "CFS_code" in config_dict:
        CFS_code_fn = config_dict["CFS_code"]
        with open(CFS_code_fn, "r") as f:
            dic_values["CFS_code_placeholder"] = f.read()
    else:
        dic_values["CFS_code_placeholder"] = ""

    generate_input_seissol_dr.generate(mode, dic_values)
