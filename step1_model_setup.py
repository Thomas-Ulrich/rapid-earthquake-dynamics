#!/usr/bin/env python3
from dynworkflow import (
    get_usgs_finite_fault_data,
    infer_fault_mesh_size_and_spatial_zoom,
    modify_FL33_34_fault_instantaneous_slip,
    generate_mesh,
    generate_input_seissol_fl33,
    prepare_velocity_model_files,
    generate_input_seissol_dr,
    generate_waveform_config_from_usgs,
)

import argparse
import os
import shutil
import sys
import glob
import subprocess

# Append kinematic_models and external folders to path
# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

relative_paths = [
    "dynworkflow/kinematic_models",
    "external",
]
for relative_path in relative_paths:
    absolute_path = os.path.join(current_script_dir, relative_path)
    if absolute_path not in sys.path:
        sys.path.append(absolute_path)


import generate_FL33_input_files
import compute_moment_rate_from_finite_fault_file
import generate_fault_output_from_fl33_input_files
import project_fault_tractions_onto_asagi_grid
import vizualizeBoundaryConditions


def run_step1():
    # Check if 'pumgen' is available
    status = os.system("which pumgen > /dev/null 2>&1")
    if status != 0:
        print("pumgen is not available.")
        sys.exit(1)

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
        if os.path.exists(finite_fault_fn):
            # absolute path given
            shutil.copy(finite_fault_fn, "tmp")
        else:
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
    suffix, ext = os.path.splitext(os.path.basename(finite_fault_fn))
    if not ext == ".txt":
        print("using USGS 1D velocity model")
        prepare_velocity_model_files.generate_usgs_velocity_files()
    else:
        print("using slipnear 1D velocity model")
        prepare_velocity_model_files.generate_slipnear_velocity_files()

    modify_FL33_34_fault_instantaneous_slip.update_file(
        f"yaml_files/FL33_34_fault.yaml"
    )
    generate_mesh.generate(h_domain=20e3, h_fault=fault_mesh_size, interactive=False)

    result = os.system("pumgen -s msh4 tmp/mesh.msh")
    if result != 0:
        sys.exit(1)

    generate_input_seissol_fl33.generate()
    compute_moment_rate_from_finite_fault_file.compute(
        finite_fault_fn, "yaml_files/material.yaml", projection
    )
    if not os.path.exists("output"):
        os.makedirs("output")

    print("step1 completed")
    return folder_name


def select_station_and_download_waveforms():
    vizualizeBoundaryConditions.generate_boundary_file("tmp/mesh.xdmf", "faults")
    # mv to tmp
    files = glob.glob("mesh_bc_faults.*")
    for file in files:
        shutil.move(file, os.path.join("tmp", os.path.basename(file)))

    generate_fault_output_from_fl33_input_files.generate(
        "tmp/mesh_bc_faults.xdmf",
        "yaml_files/FL33_34_fault.yaml",
        "output/dyn-kinmod-fault",
        "Gaussian",
        0.5,
    )
    generate_waveform_config_from_usgs.generate_waveform_config_file(
        ignore_source_files=True
    )
    command = [
        os.path.join(
            current_script_dir,
            "submodules/seismic-waveform-factory/scripts/select_stations.py",
        ),
        "waveforms_config.ini",
        "14",
        "7",
    ]
    subprocess.run(command, check=True)
    print(
        "done selecting stations. If you are not satisfied, change waveforms_config.ini and rerun:"
    )
    scommand = " ".join(command)
    print(f"{scommand}")


if __name__ == "__main__":
    folder_name = run_step1()
    select_station_and_download_waveforms()
    print(f"cd {folder_name}")
