#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from seismic_waveform_factory.geo.select_stations import select_stations

from dynworkflow import (
    generate_input_seissol_fl33,
    generate_mesh,
    generate_waveform_config_from_usgs,
    get_repo_info,
    get_usgs_finite_fault_data,
    infer_fault_mesh_size_and_spatial_zoom,
    modify_FL33_34_fault_instantaneous_slip,
    prepare_velocity_model_files,
    step1_args,
    vizualizeBoundaryConditions,
)
from kinematic_models import (
    compute_moment_rate_from_finite_fault_file,
    generate_fault_output_from_fl33_input_files,
    generate_FL33_input_files,
)


def is_slipnear_file(fn):
    with open(fn, "r") as file:
        first_line = file.readline().strip()
        return "RECTANGULAR DISLOCATION MODEL" in first_line


def copy_files(overwrite_files, setup_dir):
    yaml_dir = os.path.join(setup_dir, "yaml_files")
    nc_dir = os.path.join(setup_dir, "ASAGI_files")
    tmp_dir = os.path.join(setup_dir, "tmp")

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
            elif ext == ".txt":
                dest = Path(tmp_dir) / path.name
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

    fault_ref_args = list(map(float, args.fault_reference.split(",")))
    assert len(fault_ref_args) == 4
    assert int(fault_ref_args[3]) in [0, 1]

    custom_setup_files = [os.path.abspath(file) for file in args.custom_setup_files]

    vel_model = args.velocity_model
    if vel_model not in ["auto", "usgs"]:
        vel_model = os.path.abspath(vel_model)

    processed_MRFs = []
    for mrf in args.reference_moment_rate_functions:
        if mrf[0] == "auto":
            processed_MRFs.append(mrf)
        else:
            mrf_file, mrf_label = mrf
            mrf_file = os.path.abspath(mrf_file)
            processed_MRFs.append((mrf_file, mrf_label))

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

    fault_receiver_file = None
    if args.fault_receiver_file:
        fault_receiver_file = str(Path(args.fault_receiver_file).resolve())

    template_folder = None
    if args.template_folder:
        template_folder = str(Path(args.template_folder).resolve())

    if args.hypocenter not in ["usgs", "finite_fault"]:
        hypocenter = [float(v) for v in args.hypocenter.strip().split(",")]
        assert len(hypocenter) == 3
        derived_config["hypocenter"] = hypocenter

    allowed_gof_components = {
        "slip_distribution",
        "teleseismic_body_wf",
        "teleseismic_surface_wf",
        "regional_wf",
        "moment_rate_function",
        "fault_offsets",
        "seismic_moment",
        "slip_rate",
    }
    gof_components = [v.strip() for v in args.gof_components.strip().split(",")]
    for comp in gof_components:
        comp_name = comp.split()[0]
        if comp_name not in allowed_gof_components:
            raise ValueError(
                f"gof_component: {comp_name} not in {allowed_gof_components}"
            )

    os.chdir(derived_config["folder_name"])

    if args.fault_receiver_file:
        derived_config["fault_output_type"] = 5
        derived_config["fault_receiver_file"] = shutil.copy(fault_receiver_file, "tmp")
    else:
        derived_config["fault_output_type"] = 4
        derived_config["fault_receiver_file"] = ""

    if args.template_folder:
        os.makedirs("templates", exist_ok=True)
        for tmpl_file in glob.glob(os.path.join(template_folder, "*.tmpl.*")):
            shutil.copy2(tmpl_file, "templates")

    input_config = vars(args)
    save_config(input_config, "input_config.yaml")

    for kkk, MRF_pair in enumerate(processed_MRFs):
        if MRF_pair[0] == "auto":
            if finite_fault_model == "usgs":
                processed_MRFs[kkk] = ["tmp/moment_rate.mr", "USGS"]
            else:
                processed_MRFs[kkk] = [
                    "tmp/moment_rate_from_finite_source_file.txt",
                    "finite source model",
                ]
        else:
            mrf_file, label = MRF_pair
            copied_file = shutil.copy(mrf_file, "tmp")
            processed_MRFs[kkk] = [copied_file, label]

    derived_config["reference_STFs"] = processed_MRFs

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
            h_domain=20e3,
            h_fault=fault_mesh_size,
            interactive=False,
            vertex_union_tolerance=args.gmsh_vertex_union_tolerance,
        )
        result = os.system("pumgen -s msh4 tmp/mesh.msh")
        if result != 0:
            sys.exit(1)
        mesh_file = "tmp/mesh.puml.h5"
    else:
        mesh_file = shutil.copy(args.mesh, "tmp")
        mesh_xdmf_file = args.mesh.split("puml.h5")[0] + ".xdmf"
        shutil.copy(mesh_xdmf_file, "tmp")

    derived_config |= {
        "mesh_file": mesh_file,
        "spatial_zoom": spatial_zoom,
        "fault_mesh_size": fault_mesh_size,
        "number_of_segments": number_of_segments,
    }
    save_config(derived_config, "derived_config.yaml")

    generate_input_seissol_fl33.generate()
    compute_moment_rate_from_finite_fault_file.compute(
        finite_fault_fn, "yaml_files/material.yaml", projection, tmax=args.tmax
    )

    file_path = "tmp/moment_rate_from_finite_source_file.txt"
    if os.path.exists(file_path) and os.path.getsize(file_path) == 0:
        # todo: update for possibility of csv MRF
        print(f"{file_path} is empty (static solution?).")
        for kkk, MRF_pair in enumerate(processed_MRFs):
            assert processed_MRFs[kkk][0] != file_path
        assert args.hypocenter != "finite_fault"
        moment_rate = np.loadtxt(processed_MRFs[0][0], skiprows=2)
        with open(file_path, "w") as f:
            np.savetxt(f, moment_rate, fmt="%g")
        print("done copying refMRF to {file_path}")

    os.makedirs("output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    copy_files(custom_setup_files, ".")

    print("step1 completed")
    return derived_config["folder_name"]


def select_station_and_download_waveforms():
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
        select_stations(
            config_file="waveforms_config_regional.yaml",
            number_stations=14,
            closest_stations=7,
            distance_range=None,
            channel="*",
            store_format="mseed",
            azimuthal=False,
            station_kind="regional",
        )
        print(
            "Done selecting stations. If you are not satisfied, change "
            "waveforms_config_regional.yaml"
        )
    if teleseismic_stations == "auto":
        select_stations(
            config_file="waveforms_config_teleseismic.yaml",
            number_stations=10,
            closest_stations=0,
            distance_range=None,
            channel="B*",
            store_format="mseed",
            azimuthal=True,
            station_kind="global",
        )
    if mesh_file != "auto":
        print("custom mesh: did you think of placing receiver according to topography?")


if __name__ == "__main__":
    folder_name = run_step1()
    select_station_and_download_waveforms()
    print(f"cd {folder_name}")
