#!/usr/bin/env python3
from dynworkflow import (
    get_usgs_finite_fault_data,
    infer_fault_mesh_size_and_spatial_zoom,
    modify_FL33_34_fault_instantaneous_slip,
    generate_mesh,
    generate_input_seissol_fl33,
    prepare_velocity_model_files,
    generate_waveform_config_from_usgs,
)

import argparse
import os
import shutil
import sys
import glob
import subprocess
import numpy as np

# Append finite_fault_models and external folders to path
# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

relative_paths = [
    "dynworkflow/finite_fault_models",
    "external",
]
for relative_path in relative_paths:
    absolute_path = os.path.join(current_script_dir, relative_path)
    if absolute_path not in sys.path:
        sys.path.append(absolute_path)


import generate_FL33_input_files
import compute_moment_rate_from_finite_fault_file
import generate_fault_output_from_fl33_input_files
import vizualizeBoundaryConditions


def is_slipnear_file(fn):
    with open(fn, "r") as file:
        first_line = file.readline().strip()
        return "RECTANGULAR DISLOCATION MODEL" in first_line


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
        "event_id",
        help="usgs earthquake code or event dictionnary (dtgeo workflow)",
    )
    parser.add_argument(
        "--finite_fault_model",
        nargs=1,
        help="input filename of alternative model to usgs (e.g. Slipnear)",
        type=str,
        default=["usgs"],
    )
    parser.add_argument(
        "--tmax",
        nargs=1,
        help="remove fault slip for t_rupt> tmax",
        type=float,
        default=[None],
    )

    parser.add_argument(
        "--reference_moment_rate_function",
        nargs=1,
        help=""" Specify a reference moment rate function (for DR model ranking)
        - auto: if usgs, will download and use the STF, else moment rate inferred from the finite fault model file.
        - Alternatively, provide a STF in usgs format (2 lines of header).""",
        type=str,
        default=["auto"],
    )

    parser.add_argument(
        "--velocity_model",
        nargs=1,
        help="""Specify the velocity model:
        - auto: same as option usgs, but use the Slipnear velocity model for a Slipnear kinematic model.
        - usgs: Read the velocity model from the usgs finite fault model FSP file.
        - Alternatively, provide a velocity model in Axitra format.""",
        type=str,
        default=["auto"],
    )
    parser.add_argument(
        "--projection",
        nargs=1,
        help="""Specify the projection:
        - auto: custom transverse mercator centered roughly on the hypocenter.
        - Alternatively, provide a projection in proj4 format""",
        type=str,
        default=["auto"],
    )

    args = parser.parse_args()
    vel_model = args.velocity_model[0]
    if vel_model not in ["auto", "usgs"]:
        vel_model = os.path.abspath(vel_model)
    refMRF = args.reference_moment_rate_function[0]
    if refMRF not in ["auto"]:
        refMRF = os.path.abspath(refMRF)
    finite_fault_model = args.finite_fault_model[0]

    suffix = ""
    if finite_fault_model != "usgs":
        finite_fault_model = os.path.abspath(finite_fault_model)
        suffix, _ = os.path.splitext(os.path.basename(finite_fault_model))
        if is_slipnear_file(finite_fault_model) and vel_model == "auto":
            vel_model = "slipnear"
        elif vel_model == "auto":
            vel_model = "usgs"
    else:
        if vel_model == "auto":
            vel_model = "usgs"

    folder_name = get_usgs_finite_fault_data.get_data(
        args.event_id,
        min_magnitude=6,
        suffix=suffix,
        use_usgs_finite_fault=(finite_fault_model == "usgs"),
        download_usgs_fsp=(vel_model == "usgs"),
    )
    os.chdir(folder_name)

    refMRFfile = ""
    print(refMRF)
    if refMRF == "auto":
        if finite_fault_model == "usgs":
            refMRFfile = "tmp/moment_rate.mr"
        else:
            refMRFfile = "tmp/moment_rate_from_finite_source_file.txt"
    elif os.path.exists(refMRF):
        # test loading
        np.loadtxt(refMRF, skiprows=2)
        refMRFfile = os.path.join("tmp", refMRF)
        refMRFfile = shutil.copy(refMRF, "tmp")
    else:
        raise FileNotFoundError(f"{refMRF} does not exists")

    with open("tmp/reference_STF.txt", "w") as f:
        f.write(refMRFfile)

    projection = args.projection[0]
    if projection == "auto":
        with open("tmp/projection.txt", "r") as fid:
            projection = fid.read()
    else:
        with open("tmp/projection.txt", "w") as f:
            f.write(projection)

    if finite_fault_model != "usgs":
        finite_fault_fn = shutil.copy(finite_fault_model, "tmp")
    else:
        finite_fault_fn = "tmp/basic_inversion.param"

    (
        spatial_zoom,
        fault_mesh_size,
    ) = infer_fault_mesh_size_and_spatial_zoom.infer_quantities(
        finite_fault_fn, projection
    )

    with open("tmp/inferred_spatial_zoom.txt", "w") as f:
        f.write(str(spatial_zoom))

    with open("tmp/inferred_fault_mesh_size.txt", "w") as f:
        f.write(str(fault_mesh_size))

    generate_FL33_input_files.main(
        finite_fault_fn,
        "cubic",
        spatial_zoom,
        projection,
        write_paraview=False,
        PSRthreshold=0.0,
        tmax=args.tmax[0],
    )

    modify_FL33_34_fault_instantaneous_slip.update_file("yaml_files/FL33_34_fault.yaml")

    if vel_model == "slipnear":
        print("using slipnear 1D velocity model")
        prepare_velocity_model_files.generate_arbitrary_velocity_files()
    elif vel_model == "usgs":
        print("using USGS 1D velocity model")
        prepare_velocity_model_files.generate_usgs_velocity_files()
    else:
        print("using user-defined velocity model 1D velocity model")
        shutil.copy(vel_model, "tmp")
        prepare_velocity_model_files.generate_arbitrary_velocity_files(vel_model)

    generate_mesh.generate(h_domain=20e3, h_fault=fault_mesh_size, interactive=False)

    result = os.system("pumgen -s msh4 tmp/mesh.msh")
    if result != 0:
        sys.exit(1)

    generate_input_seissol_fl33.generate()
    compute_moment_rate_from_finite_fault_file.compute(
        finite_fault_fn, "yaml_files/material.yaml", projection, tmax=args.tmax[0]
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
