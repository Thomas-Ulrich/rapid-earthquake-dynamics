#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import itertools
import os
import random
import re
import shutil

import jinja2
import numpy as np
import pandas as pd
import seissolxdmf as sx
import yaml
from pyproj import Transformer
from scipy.stats import qmc
from scipy import integrate

from dynworkflow import step1_args
from dynworkflow.estimate_nucleation_radius import compute_critical_nucleation


def infer_duration(time, moment_rate):
    # duplicate from compile_scenario_macro_properties.py
    moment = integrate.cumulative_trapezoid(moment_rate, time, initial=0)
    M0 = np.trapz(moment_rate[:], x=time[:])
    return np.amax(time[moment < 0.99 * M0])


def compute_max_slip(fn):
    sx0 = sx.seissolxdmf(fn)
    ndt = sx0.ReadNdt()
    ASl = sx0.ReadData("ASl", ndt - 1)
    if np.any(np.isnan(ASl)):
        ASl = sx0.ReadData("ASl", ndt - 2)
    max_slip = ASl.max()
    assert max_slip > 0
    return max_slip


def parse_parameter_string(param_str):
    param_str = param_str.replace("_", "")
    input_config = {}
    for match in re.finditer(r"(\w+)=([^\s]+)", param_str):
        key, val = match.group(1), match.group(2)
        if key == "cohesion":
            input_config["cohesion"] = [
                list(map(float, pair.split(","))) for pair in val.split(";")
            ]
        else:
            input_config[key] = [float(v) for v in val.split(",") if v.strip()]
    print(input_config)
    return input_config


def render_file(templateEnv, template_par, template_fname, out_fname, verbose=True):
    template = templateEnv.get_template(template_fname)
    outputText = template.render(template_par)
    with open(out_fname, "w") as fid:
        fid.write(outputText)
    if verbose:
        print(f"done creating {out_fname}")


def generate_param_df(input_config, number_of_segments, first_simulation_id):
    mode = input_config["mode"]
    parameters_structured = input_config["parameters_structured"]
    names = sorted(parameters_structured.keys())
    names_no_cohesion = [name for name in names if name != "cohesion"]

    if ("C" not in names) and ("dc" not in names):
        raise ValueError("nor C nor d_c given in parameters")
    if ("C" in names) and ("d_c" in names):
        raise ValueError("both C and d_c given in parameters")

    cohesion_values = parameters_structured["cohesion"]
    cohesion_ids = list(range(len(cohesion_values)))

    if mode == "latin_hypercube":
        bounds = {}
        for name in names_no_cohesion:
            values = parameters_structured[name]
            bounds[name] = (min(values), max(values))

        l_bounds = [bounds[name][0] for name in names_no_cohesion]
        u_bounds = [bounds[name][1] for name in names_no_cohesion]

        # cohesion is fixed in this appraoch
        cohesion_values = parameters_structured["cohesion"]
        assert len(cohesion_values) == 1

        if not os.path.exists("tmp/seed.txt"):
            seed = random.randint(1, 1000000)
            # keep the seed for reproducibility
            with open("tmp/seed.txt", "w+") as fid:
                fid.write(f"{seed}\n")
        else:
            with open("tmp/seed.txt", "r") as fid:
                seed = int(fid.readline())
            print("seed read from tmp/seed.txt")

        sampler = qmc.LatinHypercube(d=3, seed=seed)
        nsample = input_config["nsamples"]
        sample = sampler.random(n=nsample)
        pars = qmc.scale(sample, l_bounds, u_bounds)
        pars = np.around(pars, decimals=3)
        column_of_zeros = np.zeros((1, nsample))
        pars = np.insert(pars, 0, column_of_zeros, axis=1)
        labels = names_no_cohesion

    elif mode == "grid_search":
        use_R_segment_wise = False

        if use_R_segment_wise:
            Rvalues = np.array(parameters_structured["R"]).reshape(
                (number_of_segments, -1)
            )
            labels = [name for name in names_no_cohesion if name != "R"]
            for i in range(number_of_segments):
                labels.append(f"R_{i + 1}")
                parameters_structured[f"R_{i + 1}"] = Rvalues[i, :]
            parameters_structured["cohesion_idx"] = cohesion_ids
            labels = ["cohesion_idx"] + labels
            params = [parameters_structured[name] for name in labels]
        else:
            labels = ["cohesion_idx"] + names_no_cohesion.copy()
            parameters_structured["cohesion_idx"] = cohesion_ids
            params = [parameters_structured[name] for name in labels]

        # Generate all combinations of parameter values
        param_combinations = list(itertools.product(*params))

        # Convert combinations to numpy array and round to desired decimals
        pars = np.around(np.array(param_combinations), decimals=3)
    elif mode == "picked_models":
        n = len(cohesion_values)
        for name in names:
            assert len(parameters_structured[name]) == n
        labels = ["cohesion_idx"] + names_no_cohesion.copy()
        parameters_structured["cohesion_idx"] = cohesion_ids
        pars = [[parameters_structured[name][i] for name in labels] for i in range(n)]
        pars = np.array(pars)
    else:
        raise NotImplementedError(f"unkown mode {mode}")

    param_df = pd.DataFrame(pars, columns=labels)
    param_df["cohesion_idx"] = param_df["cohesion_idx"].astype(int)
    param_df["cohesion_value"] = param_df["cohesion_idx"].apply(
        lambda i: tuple(cohesion_values[int(i)])
    )
    param_df.index += first_simulation_id

    print(param_df)
    return param_df


def compute_fault_sampling(kinmod_duration):
    if kinmod_duration < 15:
        return 0.25
    elif kinmod_duration < 30:
        return 0.5
    elif kinmod_duration < 60:
        return 1.0
    elif kinmod_duration < 200:
        return 2.5
    else:
        return 5.0


def generate_R_yaml_block(Rvalues):
    if len(Rvalues) == 1:
        return f"""        [R]: !ConstantMap
            map:
              R: {Rvalues[0]}"""

    R_yaml_block = """        [R]: !Any
           components:"""
    for p, Rp in enumerate(Rvalues):
        fault_id = 3 if p == 0 else 64 + p
        R_yaml_block += f"""
            - !GroupFilter
              groups: {fault_id}
              components: !ConstantMap
                 map:
                   R: {Rp}"""
    return R_yaml_block


def extract_template_params(
    i,
    row,
    cohesion_values,
    hypo,
    max_slip,
    input_config,
    derived_config,
):
    if "C" in row:
        Cname = "C"
        C = row[Cname]
        d_c = f'{C} * math.max({0.15 * max_slip}, x["fault_slip"])'
    else:
        Cname = "dc"
        C = row[Cname]
        d_c = str(C)

    cohi = int(row["cohesion_idx"])
    B = row["B"]

    # Extract values for all keys matching "R", "R_1", "R_2", etc.
    R_values = row[row.index.str.match(r"^R(_\d+)?$")].tolist()
    cohesion_const, cohesion_lin, cohesion_depth = cohesion_values[cohi]

    template_param = {
        "cohesion_const": cohesion_const * 1e6,
        "cohesion_lin": cohesion_lin * 1e6,
        "cohesion_depth": cohesion_depth * 1e3,
        "B": B,
        "d_c": d_c,
        "hypo_x": hypo[0],
        "hypo_y": hypo[1],
        "hypo_z": hypo[2],
        "mesh_file": derived_config["mesh_file"],
    }
    if len(R_values) > 0:
        template_param["R_yaml_block"] = generate_R_yaml_block(R_values)
        sR = "_".join(map(str, R_values))
    if "mus" in row:
        template_param["mu_s"] = row["mus"]
    if "mud" in row:
        template_param["mu_d"] = row["mud"]
    if "sigman" in row:
        sigma_n = row["sigman"]
        template_param["sigma_n"] = f"-{sigma_n}e6"
        row = row.rename({"sigman": "sn"})

        # Remove R-related keys
    row_cleaned = row.drop(row.index[row.index.str.match(r"^R(_\d+)?$")]).drop(
        ["cohesion_value", "cohesion_idx"]
    )

    code = f"{i:04}_coh{cohesion_const}_{cohesion_lin}_" + "_".join(
        [f"{var}{val}" for var, val in row_cleaned.items()]
    )
    if len(R_values) > 0:
        code += f"_R{sR}"
    return template_param, code


def generate():
    # load first default arguments for backwards compatibility
    args = step1_args.get_args()
    input_config = vars(args)

    with open("input_config.yaml", "r") as f:
        input_config |= yaml.safe_load(f)

    if not os.path.exists("yaml_files"):
        os.makedirs("yaml_files")

    fl33_file_candidates = [
        "output/dyn-kinmod-fault.xdmf",
        "extracted_output/dyn-kinmod_extracted-fault.xdmf",
        "extracted_output/dyn-kinmod_compacted-fault.xdmf",
        "output_fl33/fl33-fault.xdmf",
        "output/fl33-fault.xdmf",
        "extracted_output/fl33_extracted-fault.xdmf",
        "extracted_output/fl33_compacted-fault.xdmf",
    ]

    try:
        kinmod_fn = next(f for f in fl33_file_candidates if os.path.exists(f))
    except StopIteration:
        raise FileNotFoundError(
            (
                "None of the dyn-kinmod or fl33-fault.xdmf files were found "
                "in the expected directories."
            )
        )

    max_slip = compute_max_slip(kinmod_fn)

    # Get the directory of the script
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    input_file_dir = f"{script_directory}/input_files"

    if "template_folder" in input_config.keys():
        search_path = ["templates", input_file_dir]
    else:
        search_path = input_file_dir

    templateLoader = jinja2.FileSystemLoader(searchpath=search_path)
    templateEnv = jinja2.Environment(loader=templateLoader)

    parameters_structured = parse_parameter_string(input_config["parameters"])
    input_config["parameters_structured"] = parameters_structured
    cohesion_values = parameters_structured["cohesion"]

    with open("derived_config.yaml", "r") as f:
        derived_config = yaml.safe_load(f)
    number_of_segments = derived_config["number_of_segments"]
    mode = input_config["mode"]

    longer_and_more_frequent_output = mode == "picked_models"

    first_sim_id = derived_config["first_simulation_id"]
    param_df = generate_param_df(input_config, number_of_segments, first_sim_id)
    param_df.to_csv(
        f"simulation_parameters_{first_sim_id}.csv", index=True, index_label="id"
    )
    nsample = len(param_df)
    derived_config["simulation_batch_size"] = nsample
    print(f"parameter space has {nsample} samples")

    projection = derived_config["projection"]
    transformer = Transformer.from_crs("epsg:4326", projection, always_xy=True)
    # [:] required to get a copy
    hypo = derived_config["hypocenter"][:]
    hypo[2] *= -1e3
    hypo[0], hypo[1] = transformer.transform(hypo[0], hypo[1])

    fn_mr = "tmp/moment_rate_from_finite_source_file.txt"
    moment_rate = np.loadtxt(fn_mr)
    kinmod_duration = infer_duration(moment_rate[:, 0], moment_rate[:, 1])
    fault_sampling = compute_fault_sampling(kinmod_duration)

    list_fault_yaml = []
    if "mud" in parameters_structured.keys():
        fn_fault_template = "fault_constant_friction.tmpl.yaml"
    else:
        fn_fault_template = "fault.tmpl.yaml"
    print(fn_fault_template)

    for idx, row in param_df.iterrows():
        template_par, code = extract_template_params(
            idx,
            row,
            cohesion_values,
            hypo,
            max_slip,
            input_config,
            derived_config,
        )
        template_par["r_crit"] = 3000.0
        fn_fault = f"yaml_files/fault_{code}.yaml"
        list_fault_yaml.append(fn_fault)
        render_file(templateEnv, template_par, fn_fault_template, fn_fault)

        if input_config["seissol_end_time"] == "auto":
            template_par["end_time"] = kinmod_duration + max(
                20.0, 0.25 * kinmod_duration
            )
            use_terminator = True
        else:
            template_par["end_time"] = float(input_config["seissol_end_time"])
            use_terminator = False

        terminator = input_config["terminator"].lower()
        if terminator != "auto":
            use_terminator = True if terminator == "true" else False

        if longer_and_more_frequent_output:
            template_par["terminatorMomentRateThreshold"] = -1
            template_par["surface_output_interval"] = 1.0
        else:
            template_par["terminatorMomentRateThreshold"] = (
                1e17 if use_terminator else -1
            )
            template_par["surface_output_interval"] = 5.0

        if input_config["regional_synthetics_generator"] == "seissol":
            template_par["enable_receiver_output"] = 1
        else:
            template_par["enable_receiver_output"] = 0

        template_par["fault_fname"] = fn_fault
        template_par["output_file"] = f"output/dyn_{code}"
        template_par["material_fname"] = "yaml_files/material.yaml"
        template_par["fault_print_time_interval"] = fault_sampling

        fault_ref_args = list(map(float, input_config["fault_reference"].split(",")))
        ref_x, ref_y, ref_z, ref_method = fault_ref_args
        template_par["ref_x"] = ref_x
        template_par["ref_y"] = ref_y
        template_par["ref_z"] = ref_z
        template_par["ref_method"] = int(ref_method)
        template_par["fault_receiver_file"] = derived_config["fault_receiver_file"]
        template_par["fault_output_type"] = derived_config["fault_output_type"]

        fn_param = f"parameters_dyn_{code}.par"
        render_file(templateEnv, template_par, "parameters_dyn.tmpl.par", fn_param)

    fnames = ["fault_slip.yaml"]
    for fn in fnames:
        shutil.copy(f"{input_file_dir}/{fn}", f"yaml_files/{fn}")

    fl33_file_candidates = [
        "output_fl33/fl33-fault.xdmf",
        "output/fl33-fault.xdmf",
        "extracted_output/fl33_extracted-fault.xdmf",
        "extracted_output/fl33_compacted-fault.xdmf",
    ]

    try:
        fl33_file = next(f for f in fl33_file_candidates if os.path.exists(f))
    except StopIteration:
        raise FileNotFoundError(
            (
                "None of the fl33-fault.xdmf files were found "
                "in the expected directories."
            )
        )

    list_nucleation_size = compute_critical_nucleation(
        fl33_file,
        "yaml_files/material.yaml",
        "yaml_files/fault_slip.yaml",
        list_fault_yaml,
        hypo,
    )
    print(list_nucleation_size)

    parameter_files = []
    for k, (idx, row) in enumerate(param_df.iterrows()):
        template_par, code = extract_template_params(
            idx,
            row,
            cohesion_values,
            hypo,
            max_slip,
            input_config,
            derived_config,
        )
        fn_fault = f"yaml_files/fault_{code}.yaml"
        fn_param = f"parameters_dyn_{code}.par"
        if list_nucleation_size[k]:
            template_par["r_crit"] = list_nucleation_size[k]
            render_file(templateEnv, template_par, fn_fault_template, fn_fault)
            parameter_files.append(fn_param)
        else:
            print(f"removing {fn} and {fn_param} (nucleation too large)")
            os.remove(fn)
            os.remove(fn_param)

    # write parts.txt files
    nfiles = len(parameter_files)
    n = 1
    parts = np.array_split(np.arange(nfiles), n)
    split_files = [list(np.array(parameter_files)[part]) for part in parts]
    for i, part in enumerate(split_files):
        part_filename = f"part_{i + 1}.txt"
        with open(part_filename, "w") as f:
            for par_file in part:
                f.write(par_file + "\n")
        print(f"done writing {part_filename}")

    derived_config["first_simulation_id"] += len(param_df)
    with open("derived_config.yaml", "w") as f:
        yaml.dump(derived_config, f)
