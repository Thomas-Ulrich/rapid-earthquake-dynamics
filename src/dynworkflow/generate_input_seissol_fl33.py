#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import glob
import os
import re

import jinja2
import numpy as np
import yaml
from scipy.spatial.distance import pdist


def generate():
    with open("derived_config.yaml", "r") as f:
        derived_config = yaml.safe_load(f)
    with open("input_config.yaml", "r") as f:
        input_config = yaml.safe_load(f)

    # Get the directory of the script
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    input_file_dir = f"{script_directory}/input_files"

    if "template_folder" in input_config.keys():
        input_file_dir = ["templates", input_file_dir]

    templateLoader = jinja2.FileSystemLoader(searchpath=input_file_dir)
    templateEnv = jinja2.Environment(loader=templateLoader)

    # Regular expression patterns to match vertex lines
    vertex_pattern = re.compile(r"VRTX (\d+) ([\d.-]+) ([\d.-]+) ([\d.-]+)")
    allv = []
    ts_files = sorted(glob.glob("tmp/*.ts"))

    for i, fn in enumerate(ts_files):
        vertices = []
        with open(fn, "r") as file:
            for line in file:
                match = vertex_pattern.match(line)
                if match:
                    vertex_id, x, y, z = map(float, match.groups())
                    vertices.append([x, y, z])
        allv.extend(vertices)
    xyz = np.array(allv)

    # Find the maximum distance
    max_distance = np.max(pdist(xyz))

    # 3200 is the smallest wave speed in the model
    end_time = max_distance / 3200.0
    template_par = {}
    # well in theory we would need to run for end_time, but practically
    # a portion of it may be sufficient
    end_time = max(30.0, 0.6 * end_time)
    template_par["end_time"] = end_time
    derived_config["pseudo_static_simulation_end_time"] = end_time

    with open("derived_config.yaml", "w") as f:
        yaml.dump(derived_config, f)

    template_par["material_fname"] = "yaml_files/material.yaml"

    fault_ref_args = list(map(float, input_config["fault_reference"].split(",")))
    ref_x, ref_y, ref_z, ref_method = fault_ref_args
    template_par["ref_x"] = ref_x
    template_par["ref_y"] = ref_y
    template_par["ref_z"] = ref_z
    template_par["ref_method"] = int(ref_method)

    mesh_file = derived_config["mesh_file"]
    template_par["mesh_file"] = mesh_file

    template = templateEnv.get_template("parameters_fl34.tmpl.par")
    outputText = template.render(template_par)
    fname = "parameters_fl33.par"
    with open(fname, "w") as fid:
        fid.write(outputText)
    print(f"done creating {fname}")


if __name__ == "__main__":
    generate()
