#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

from dynworkflow import (
    get_usgs_finite_fault_data,
    infer_fault_mesh_size_and_spatial_zoom,
    modify_FL33_34_fault_instantaneous_slip,
    generate_mesh,
    generate_input_seissol_fl33,
    prepare_velocity_model_files,
    generate_waveform_config_from_usgs,
    vizualizeBoundaryConditions,
    get_repo_info,
    step1_args,
)
from kinematic_models import (
    generate_FL33_input_files,
    compute_moment_rate_from_finite_fault_file,
    generate_fault_output_from_fl33_input_files,
)

import argparse
import os
import shutil
import sys
import glob
import subprocess
import numpy as np
import yaml
from pathlib import Path


def is_slipnear_file(fn):
    with open(fn, "r") as file:
        first_line = file.readline().strip()
        return "RECTANGULAR DISLOCATION MODEL" in first_line


def copy_files(overwrite_files, setup_dir):
    yaml_dir = os.path.join(setup_dir, "yaml_files")
    nc_dir = os.path.join(setup_dir, "ASAGI_files")

    os.makedirs(yaml_dir, exist_ok=True)
    os.makedirs(nc_dir, exist_ok=True)

    for path_str in overwrite_files:
        path = Path(path_str).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File or directory not found: {path}")

        if path.is_file():
            ext = path.suffix.lower()
            if ext == ".yaml":
                dest = Path(yaml_dir) / path.name
            elif ext == ".nc":
                dest = Path(nc_dir) / path.name
            elif ext == ".csv":
                # offset file
                dest = Path(setup_dir) / path.name
            else:
                raise ValueError(f"Skipping unsupported file type: {path}")

            print(f"Copying file {path} to {dest}")
            shutil.copy2(path, dest)

        elif path.is_dir():
            dest = Path(setup_dir) / path.name
            if dest.exists():
                print(f"Removing existing folder before copy: {dest}")
                shutil.rmtree(dest)
            print(f"Copying folder {path} to {dest}")
            shutil.copytree(path, dest)

        else:
            raise ValueError(f"Unsupported path type: {path}")


def process_parser():
    args = step1_args.get_args()

    if args.config:
        # First, keep the defaults from argparse
        args_dict = vars(args)

        # Then load YAML and update only given fields
        with open(args.config, "r") as f:
            yaml_args = yaml.safe_load(f) or {}
        print(f"Loaded config from {args.config}")

        # Update args_dict with fields from YAML
        args_dict.update(yaml_args)

        # Create updated Namespace
        args = argparse.Namespace(**args_dict)

    else:
        args_dict = vars(args)

    print("Using parameters:")
    for k, v in args_dict.items():
        print(f"  {k}: {v}")

    return args


def save_config(args_dict, config_out):
    with open(config_out, "w") as f:
        yaml.dump(args_dict, f)
    print(f"Saved config to {config_out}")


def run_step1():
    # Check if 'pumgen' is available
    status = os.system("which pumgen > /dev/null 2>&1")
    if status != 0:
        print("pumgen is not available.")
        sys.exit(1)

    args = process_parser()

    assert args.terminator.lower() in ["auto", "true", "false"], (
        f"Invalid value for --terminator: {args.terminator}."
        " Must be 'auto', 'True', or 'False'."
    )
    if args.seissol_end_time != "auto":
        try:
            float(args.seissol_end_time)
        except ValueError:
            raise ValueError(
                (
                    "Invalid value for --seissol_end_time: "
                    f"'{args.seissol_end_time}'. Must be 'auto' or a float."
                )
            )

    custom_setup_files = [os.path.abspath(file) for file in args.custom_setup_files]

    if args.CFS_code:
        CFS_code = os.path.abspath(args.CFS_code)

    vel_model = args.velocity_model
    if vel_model not in ["auto", "usgs"]:
        vel_model = os.path.abspath(vel_model)
    refMRF = args.reference_moment_rate_function
    if refMRF not in ["auto"]:
        refMRF = os.path.abspath(refMRF)
    finite_fault_model = args.finite_fault_model

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

    derived_config = get_usgs_finite_fault_data.get_data(
        args.event_id,
        min_magnitude=6,
        suffix=suffix,
        use_usgs_finite_fault=(finite_fault_model == "usgs"),
        download_usgs_fsp=(vel_model == "usgs"),
        use_usgs_hypocenter=(args.hypocenter == "usgs"),
    )
    repo_info = get_repo_info.get_repo_info()
    derived_config["repository"] = repo_info

    if args.hypocenter not in ["usgs", "finite_fault"]:
        hypocenter = [float(v) for v in args.hypocenter.strip().split(",")]
        assert len(hypocenter) == 3
        derived_config["hypocenter"] = hypocenter

    allowed_gof_components = {
        "slip_distribution",
        "telsesismic_body_wf",
        "regional_wf",
        "moment_rate_function",
        "fault_offsets",
        "seismic_moment",
    }
    gof_components = args.gof_components.strip().split(",").split()[0]
    for comp in gof_components:
        if comp not in allowed_gof_components:
            raise ValueError(f"gof_component: {comp} not in {allowed_gof_components}")

    os.chdir(derived_config["folder_name"])
    input_config = vars(args)
    save_config(input_config, "input_config.yaml")

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

    derived_config["reference_STF"] = refMRFfile
    derived_config["first_simulation_id"] = args.first_simulation_id
    projection = args.projection
    if projection == "auto":
        projection = derived_config["projection"]
    else:
        derived_config["projection"] = projection

    if finite_fault_model != "usgs":
        finite_fault_fn = shutil.copy(finite_fault_model, "tmp")
    else:
        finite_fault_fn = "tmp/basic_inversion.param"

    if args.fault_mesh_size != "auto":
        fault_mesh_size = float(args.fault_mesh_size)
    (
        spatial_zoom,
        fault_mesh_size,
        number_of_segments,
    ) = infer_fault_mesh_size_and_spatial_zoom.infer_quantities(
        finite_fault_fn, projection, args.fault_mesh_size
    )
    save_config(derived_config, "derived_config.yaml")

    generate_FL33_input_files.main(
        finite_fault_fn,
        "cubic",
        spatial_zoom,
        projection,
        write_paraview=False,
        PSRthreshold=0.0,
        tmax=args.tmax,
    )

    # reload config with potentially hypocenter
    with open("derived_config.yaml") as f:
        derived_config = yaml.safe_load(f)

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

    if args.mesh == "auto":
        generate_mesh.generate(
            h_domain=20e3, h_fault=fault_mesh_size, interactive=False
        )
        result = os.system("pumgen -s msh4 tmp/mesh.msh")
        if result != 0:
            sys.exit(1)
        mesh_file = "tmp/mesh.puml.h5"
    else:
        mesh_file = shutil.copy(args.mesh, "tmp")
        mesh_xdmf_file = args.mesh.split("puml.h5")[0] + ".xdmf"
        shutil.copy(mesh_xdmf_file, "tmp")

    if args.CFS_code:
        CFS_code = shutil.copy(CFS_code, "tmp")
        derived_config["CFS_code"] = CFS_code

    derived_config |= {
        "mesh_file": mesh_file,
        "spatial_zoom": spatial_zoom,
        "fault_mesh_size": fault_mesh_size,
        "mu_delta_min": input_config["mu_delta_min"],
        "mu_d": input_config["mu_d"],
        "number_of_segments": number_of_segments,
    }
    save_config(derived_config, "derived_config.yaml")

    generate_input_seissol_fl33.generate()
    compute_moment_rate_from_finite_fault_file.compute(
        finite_fault_fn, "yaml_files/material.yaml", projection, tmax=args.tmax
    )

    file_path = "tmp/moment_rate_from_finite_source_file.txt"
    if os.path.exists(file_path) and os.path.getsize(file_path) == 0:
        print(f"{file_path} is empty (static solution?).")
        assert refMRF != file_path
        assert args.hypocenter != "finite_fault"
        moment_rate = np.loadtxt(refMRF, skiprows=2)
        with open(file_path, "w") as f:
            np.savetxt(f, moment_rate, fmt="%g")
        print("done copying refMRF to {file_path}")

    if not os.path.exists("output"):
        os.makedirs("output")
    copy_files(custom_setup_files, ".")

    print("step1 completed")
    return derived_config["folder_name"]


def select_station_and_download_waveforms():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    with open("input_config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    mesh_file = config_dict["mesh"]
    regional_seismic_stations = config_dict["regional_seismic_stations"]
    teleseismic_stations = config_dict["teleseismic_stations"]
    if mesh_file == "auto":
        mesh_xdmf_file = "tmp/mesh.xdmf"
    else:
        mesh_file = "tmp/" + os.path.basename(mesh_file)
        mesh_xdmf_file = mesh_file.split("puml.h5")[0] + ".xdmf"

    mesh_prefix = os.path.basename(mesh_xdmf_file).split(".xdmf")[0]

    vizualizeBoundaryConditions.generate_boundary_file(mesh_xdmf_file, "faults")
    # mv to tmp
    files = glob.glob(f"{mesh_prefix}_bc_faults.*")
    for file in files:
        shutil.move(file, os.path.join("tmp", os.path.basename(file)))

    generate_fault_output_from_fl33_input_files.generate(
        f"tmp/{mesh_prefix}_bc_faults.xdmf",
        "yaml_files/FL33_34_fault.yaml",
        "output/dyn-kinmod-fault",
        "Gaussian",
        0.5,
    )
    generate_waveform_config_from_usgs.generate_waveform_config_file(
        regional_stations=regional_seismic_stations,
        teleseismic_stations=teleseismic_stations,
        ignore_source_files=True,
    )

    if regional_seismic_stations == "auto":
        command = [
            os.path.join(
                current_script_dir,
                "submodules/seismic-waveform-factory/scripts/select_stations.py",
            ),
            "waveforms_config_regional.ini",
            "14",
            "7",
        ]
        subprocess.run(command, check=True)
        print(
            "Done selecting stations. If you are not satisfied, change "
            "waveforms_config_regional.ini and rerun:"
        )
        scommand = " ".join(command)
        print(f"{scommand}")
    if teleseismic_stations == "auto":
        raise NotImplementedError("please specify manually teleseismic stations")


if __name__ == "__main__":
    folder_name = run_step1()
    select_station_and_download_waveforms()
    print(f"cd {folder_name}")
