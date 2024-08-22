#!/usr/bin/env python3
from dynworkflow import (
    get_usgs_finite_fault_data,
    infer_fault_mesh_size_and_spatial_zoom,
    modify_FL33_34_fault_instantaneous_slip,
    generate_mesh,
    generate_input_seissol_fl33,
    prepare_velocity_model_files,
)

import argparse
import os
import shutil
import sys

# Append kinematic_models folder to path
# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "dynworkflow/kinematic_models"
absolute_path = os.path.join(current_script_dir, relative_path)
if absolute_path not in sys.path:
    sys.path.append(absolute_path)

import generate_FL33_input_files
import compute_moment_rate_from_finite_fault_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="automatically setup a dynamic rupture model from a kinematic model"
    )
    parser.add_argument(
        "usgs_id_or_dtgeo_npy",
        help="usgs earthquake code or event dictionnary (dtgeo workflow)",
    )
    parser.add_argument(
        "--user_defined_kinematic_model",
        nargs=1,
        help="input filename of alternative model to usgs (e.g. Slipnear)",
        type=str,
    )
    args = parser.parse_args()

    suffix = ""
    if args.user_defined_kinematic_model:
        finite_fault_fn = args.user_defined_kinematic_model[0]
        suffix, ext = os.path.splitext(os.path.basename(finite_fault_fn))

    folder_name = get_usgs_finite_fault_data.get_data(
        args.usgs_id_or_dtgeo_npy, min_magnitude=7, suffix=suffix
    )
    os.chdir(folder_name)

    if args.user_defined_kinematic_model:
        shutil.copy(f"../{finite_fault_fn}", "tmp")
        finite_fault_fn = f"tmp/{finite_fault_fn}"

    if not args.user_defined_kinematic_model:
        finite_fault_fn = f"tmp/basic_inversion.param"

    with open(f"tmp/hypocenter.txt", "r") as f:
        lon, lat, _ = f.read().split()

    projection = f"+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0={lon} +lat_0={lat}"

    with open(f"tmp/projection.txt", "w") as f:
        f.write(projection)

    (
        spatial_zoom,
        fault_mesh_size,
    ) = infer_fault_mesh_size_and_spatial_zoom.infer_quantities(
        finite_fault_fn, projection
    )

    with open(f"tmp/inferred_spatial_zoom.txt", "w") as f:
        f.write(str(spatial_zoom))

    with open(f"tmp/inferred_fault_mesh_size.txt", "w") as f:
        f.write(str(fault_mesh_size))

    generate_FL33_input_files.main(
        finite_fault_fn,
        "cubic",
        spatial_zoom,
        projection,
        write_paraview=False,
        PSRthreshold=0.0,
    )
    if not args.user_defined_kinematic_model:
        prepare_velocity_model_files.generate_usgs_velocity_files()
    else:
        prepare_velocity_model_files.generate_slipnear_velocity_files()

    modify_FL33_34_fault_instantaneous_slip.update_file(
        f"yaml_files/FL33_34_fault.yaml"
    )
    generate_mesh.generate(h_domain=20e3, h_fault=fault_mesh_size, interactive=False)

    os.system("module load pumgen; pumgen -s msh4 tmp/mesh.msh")
    generate_input_seissol_fl33.generate()
    compute_moment_rate_from_finite_fault_file.compute(
        finite_fault_fn, "yaml_files/material.yaml", projection
    )
    print("step1 completed")
