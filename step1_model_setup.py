#!/usr/bin/env python3
from dynworkflow import (
    get_usgs_finite_fault_data,
    infer_fault_mesh_size_and_spatial_zoom,
    modify_FL33_34_fault_instantaneous_slip,
    generate_mesh,
    generate_input_seissol_fl33,
    prepare_velocity_model_files,
    generate_waveform_config_from_usgs,
    vizualizeBoundaryConditions,
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


def get_parser():
    parser = argparse.ArgumentParser(
        description="""
        Automatically set up an ensemble of dynamic rupture models from a kinematic
        finite fault model.

        You can either:
        1. Provide all parameters via command-line arguments, or
        2. Use the --config option to load parameters from a YAML config file.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML config file containing all input parameters.",
    )
    parser.add_argument(
        "--CFS_code",
        type=str,
        help="""
        code to add at {{ CFS_code_placeholder }} in
        dynworkflow/input_files/fault.tmpl.yaml
        (e.g. for setting-up cohesion and shear stress on receiver faults
        """,
    )
    parser.add_argument(
        "--custom_setup_files",
        type=lambda s: s.split(";"),
        default=[],
        help="""
           Semicolon-separated list of files (e.g., material.yaml;data.nc) to
           overwrite default files in the earthquake setup folder. yaml files
           will be copied to yaml_files folder. nc files to ASAGI_files folder
        """,
    )

    parser.add_argument(
        "--event_id",
        type=str,
        help="""
        Earthquake event identifier.
        - If using USGS, this is the event ID (e.g., 'us6000d3zh').
        - If using a custom model, this can be a local event dictionary name.
        """,
    )

    parser.add_argument(
        "--fault_mesh_size",
        type=str,
        default="auto",
        help="""
        auto: inferred from fault dimensions
        else provide a value
        """,
    )

    parser.add_argument(
        "--finite_fault_model",
        type=str,
        default="usgs",
        help="Path to an alternative finite fault model file.",
    )

    parser.add_argument(
        "--hypocenter",
        type=str,
        default="finite_fault",
        help="""
        Specify the hypocenter location. Options:
        - finite_fault: use the first rupturing point in the finite-fault model.
        - usgs: use the most recent origin coordinates from the USGS.
        - lon,lat,depth_km: manually provide coordinates (e.g., '86.08,27.67,15').
        """,
    )

    parser.add_argument(
        "--mesh",
        type=str,
        default="auto",
        help="Path to an alternative mesh file",
    )

    parser.add_argument(
        "--mode",
        choices=["grid_search", "latin_hypercube", "picked_models"],
        default="grid_search",
        help="Sampling strategy for DR input generation.",
    )

    parser.add_argument(
        "--mu_d",
        type=float,
        default=0.2,
        help="dynamic friction",
    )

    parser.add_argument(
        "--mu_delta_min",
        type=float,
        default=0.01,
        help="minimum allowed mu_s - mu_d",
    )

    parser.add_argument(
        "--parameters",
        type=str,
        default=(
            "B=0.9,1.0,1.1,1.2 C=0.1,0.2,0.3,0.4,0.5 "
            "R=0.55,0.6,0.65,0.7,0.8,0.9 cohesion=0.25,1,6"
        ),
        help=(
            "Parameter definitions in 'key=val1,val2 ...' format. "
            "Separate key-value pairs with spaces. For cohesion, use "
            "semicolon-separated tuples, with 3 values K0,K1 (MPa) and d_coh (km)"
            "K(z)  = K0 + K1 max(d-d_coh/d_coh))"
            "e.g. B=0.2,0.3 C=0.1,0.2,0.3 R=0.7,0.8 cohesion=0.25,0,6;0.3,1,6"
        ),
    )

    parser.add_argument(
        "--projection",
        type=str,
        default="auto",
        help="""
        Map projection specification.
        - 'auto': transverse Mercator centered on the USGS first estimated hypocenter.
        - OR: custom projection string in Proj4 format
        (e.g., '+proj=utm +zone=33 +datum=WGS84').
        """,
    )

    parser.add_argument(
        "--reference_moment_rate_function",
        type=str,
        default="auto",
        help="""
        Reference moment rate function (used for model ranking).
        - 'auto': download STF from USGS if available, or infer from finite fault model.
        - OR: path to a 2-column STF file in USGS format (2 lines header).
        """,
    )

    parser.add_argument(
        "--regional_seismic_stations",
        type=str,
        default="auto",
        help="""
        regional seismic stations for validating the models
        - 'auto': would be automatically determined.
        - OR: list of coma separated station, e.g.
        "NC.KRP,BK.SBAR,BK.THOM,BK.DMOR,NC.KMPB,BK.HUNT,BK.ETSL,BK.HALS,BK.MNDO"
        """,
    )

    parser.add_argument(
        "--first_simulation_id",
        type=int,
        default=0,
        help="""
        first simulation id to be use in the ensemble
        """,
    )

    parser.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="""
        Maximum rupture time in seconds.
        Slip contributions with t_rupt > tmax will be ignored.
        """,
    )

    parser.add_argument(
        "--velocity_model",
        type=str,
        default="auto",
        help="""
        Velocity model to use.
        - 'auto': choose based on finite fault model (e.g., Slipnear or USGS).
        - 'usgs': extract from the USGS FSP file.
        - OR: provide a velocity model in Axitra format.
        """,
    )

    return parser


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


def dict_to_namespace(d):
    return argparse.Namespace(**d)


def process_parser():
    parser = get_parser()
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            args_dict = yaml.safe_load(f)
        print(f"Loaded config from {args.config}")
        args_dict["config"] = args.config
        args = dict_to_namespace(args_dict)
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

    if args.hypocenter not in ["usgs", "finite_fault"]:
        hypocenter = [float(v) for v in args.hypocenter.split()]
        assert len(hypocenter) == 3
        derived_config["hypocenter"] = hypocenter

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
    generate_FL33_input_files.main(
        finite_fault_fn,
        "cubic",
        spatial_zoom,
        projection,
        write_paraview=False,
        PSRthreshold=0.0,
        tmax=args.tmax,
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
        stations=regional_seismic_stations, ignore_source_files=True
    )

    if regional_seismic_stations == "auto":
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
            "Done selecting stations. If you are not satisfied, change "
            "waveforms_config.ini and rerun:"
        )
        scommand = " ".join(command)
        print(f"{scommand}")


if __name__ == "__main__":
    folder_name = run_step1()
    select_station_and_download_waveforms()
    print(f"cd {folder_name}")
