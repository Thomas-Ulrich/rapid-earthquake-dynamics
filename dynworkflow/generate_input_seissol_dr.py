#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import numpy as np
import os
import jinja2
import shutil
from dynworkflow.estimate_nucleation_radius import compute_critical_nucleation
from scipy.stats import qmc
import random
import itertools
from dynworkflow.compile_scenario_macro_properties import infer_duration
import seissolxdmf as sx
from pyproj import Transformer
import yaml
import re
import pandas as pd


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
    if "C" in input_config:
        Cname = "C"
        C_values = input_config["C"]
    elif "d_c" in input_config:
        Cname = "dc"
        C_values = input_config["d_c"]
    else:
        raise ValueError("nor C nor d_c given in parameters")
    if ("C" in input_config) and ("d_c" in input_config):
        raise ValueError("both C and d_c given in parameters")

    B_values = input_config["B"]
    R_values = input_config["R"]

    cohesion_values = input_config["cohesion"]
    cohesion_ids = list(range(len(cohesion_values)))

    if mode == "latin_hypercube":
        R0, R1 = min(R_values), max(R_values)
        B0, B1 = min(B_values), max(B_values)
        C0, C1 = min(C_values), max(C_values)

        l_bounds = [R0, B0, C0]
        u_bounds = [R1, B1, C1]

        # cohesion is fixed in this appraoch
        cohesion_values = input_config["cohesion"]
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
        labels = ["B", Cname, "R"]

    elif mode == "grid_search":
        use_R_segment_wise = False

        if use_R_segment_wise:
            params = [cohesion_ids, B_values, C_values] + [
                R_values
            ] * number_of_segments
            labels = ["cohesion_idx", "B", Cname] + [
                f"R_seg{i + 1}" for i in range(number_of_segments)
            ]
            assert len(params) == number_of_segments + 3
        else:
            params = [cohesion_ids, B_values, C_values, R_values]
            labels = ["cohesion_idx", "B", Cname, "R"]
            assert len(params) == 4

        # Generate all combinations of parameter values
        param_combinations = list(itertools.product(*params))

        # Convert combinations to numpy array and round to desired decimals
        pars = np.around(np.array(param_combinations), decimals=3)
    elif mode == "picked_models":
        cohesion_values = input_config["cohesion"]
        n = len(cohesion_values)
        assert len(B_values) == len(C_values) == len(R_values) == n
        pars = [
            [i, input_config["B"][i], input_config[Cname][i], input_config["R"][i]]
            for i in range(n)
        ]
        pars = np.array(pars)
        labels = ["cohesion_idx", "B", Cname, "R"]
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
    Cname,
    cohesion_values,
    hypo,
    max_slip,
    constant_d_c,
    input_config,
    derived_config,
    CFS_code_placeholder,
):
    cohi = int(row["cohesion_idx"])
    B = row["B"]
    C = row[Cname]
    R = row.drop(["cohesion_idx", "cohesion_value", "B", Cname]).values

    cohesion_const, cohesion_lin, cohesion_depth = cohesion_values[cohi]

    if constant_d_c:
        d_c = str(C)
    else:
        d_c = f'{C} * math.max({0.15 * max_slip}, x["fault_slip"])'

    template_param = {
        "R_yaml_block": generate_R_yaml_block(R),
        "cohesion_const": cohesion_const * 1e6,
        "cohesion_lin": cohesion_lin * 1e6,
        "cohesion_depth": cohesion_depth * 1e3,
        "B": B,
        "d_c": d_c,
        "hypo_x": hypo[0],
        "hypo_y": hypo[1],
        "hypo_z": hypo[2],
        "mu_delta_min": input_config["mu_delta_min"],
        "mesh_file": derived_config["mesh_file"],
        "CFS_code_placeholder": CFS_code_placeholder,
    }

    sR = "_".join(map(str, R))
    code = f"{i:04}_coh{cohesion_const}_{cohesion_lin}_B{B}_{Cname}{C}_R{sR}"

    return template_param, code


def generate():
    if not os.path.exists("yaml_files"):
        os.makedirs("yaml_files")

    kinmod_fn = "output/dyn-kinmod-fault.xdmf"
    if not os.path.exists(kinmod_fn):
        kinmod_fn = "extracted_output/dyn-kinmod_extracted-fault.xdmf"
    max_slip = compute_max_slip(kinmod_fn)

    # Get the directory of the script
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    input_file_dir = f"{script_directory}/input_files"
    templateLoader = jinja2.FileSystemLoader(searchpath=input_file_dir)
    templateEnv = jinja2.Environment(loader=templateLoader)

    with open("input_config.yaml", "r") as f:
        input_config = yaml.safe_load(f)
    input_config |= parse_parameter_string(input_config["parameters"])
    cohesion_values = input_config["cohesion"]

    with open("derived_config.yaml", "r") as f:
        derived_config = yaml.safe_load(f)
    number_of_segments = derived_config["number_of_segments"]
    mode = input_config["mode"]

    if derived_config["CFS_code"]:
        # useful for CFS calculation to set up, fault_tag-wise,
        # cohesion, T_s and T_d (e.g. for Mendocino)

        CFS_code_fn = derived_config["CFS_code"]
        with open(CFS_code_fn, "r") as f:
            CFS_code_placeholder = f.read()
    else:
        CFS_code_placeholder = ""

    longer_and_more_frequent_output = mode == "picked_models"

    if "C" in input_config:
        constant_d_c = False
        Cname = "C"
    else:
        constant_d_c = True
        Cname = "dc"

    first_simulation_id = derived_config["first_simulation_id"]
    param_df = generate_param_df(input_config, number_of_segments, first_simulation_id)
    param_df.to_csv("simulation_parameters.csv", index=True, index_label="id")

    nsample = len(param_df)
    print(f"parameter space has {nsample} samples")

    projection = derived_config["projection"]
    transformer = Transformer.from_crs("epsg:4326", projection, always_xy=True)
    hypo = derived_config["hypocenter"]
    hypo[2] *= -1e3
    hypo[0], hypo[1] = transformer.transform(hypo[0], hypo[1])

    fn_mr = "tmp/moment_rate_from_finite_source_file.txt"
    moment_rate = np.loadtxt(fn_mr)
    kinmod_duration = infer_duration(moment_rate[:, 0], moment_rate[:, 1])
    fault_sampling = compute_fault_sampling(kinmod_duration)

    list_fault_yaml = []

    for idx, row in param_df.iterrows():
        template_par, code = extract_template_params(
            idx,
            row,
            Cname,
            cohesion_values,
            hypo,
            max_slip,
            constant_d_c,
            input_config,
            derived_config,
            CFS_code_placeholder,
        )
        template_par["r_crit"] = 3000.0
        fn_fault = f"yaml_files/fault_{code}.yaml"
        list_fault_yaml.append(fn_fault)

        render_file(templateEnv, template_par, "fault.tmpl.yaml", fn_fault)

        if longer_and_more_frequent_output:
            template_par["terminatorMomentRateThreshold"] = -1
            template_par["surface_output_interval"] = 1.0
        else:
            template_par["terminatorMomentRateThreshold"] = 1e17
            template_par["surface_output_interval"] = 5.0
        template_par["end_time"] = kinmod_duration + max(20.0, 0.25 * kinmod_duration)
        template_par["fault_fname"] = fn_fault
        template_par["output_file"] = f"output/dyn_{code}"
        template_par["material_fname"] = "yaml_files/material.yaml"
        template_par["fault_print_time_interval"] = fault_sampling
        fn_param = f"parameters_dyn_{code}.par"
        render_file(templateEnv, template_par, "parameters_dyn.tmpl.par", fn_param)

    template_par = {"mu_d": input_config["mu_d"]}
    render_file(templateEnv, template_par, "mud.tmpl.yaml", "yaml_files/mud.yaml")

    fnames = ["fault_slip.yaml"]
    for fn in fnames:
        shutil.copy(f"{input_file_dir}/{fn}", f"yaml_files/{fn}")

    if os.path.exists("output/fl33-fault.xdmf"):
        fl33_file = "output/fl33-fault.xdmf"
    elif os.path.exists("extracted_output/fl33_extracted-fault.xdmf"):
        fl33_file = "extracted_output/fl33_extracted-fault.xdmf"
    else:
        raise FileNotFoundError(
            "The files output/fl33-fault.xdmf or "
            "extracted_output/fl33_extracted-fault.xdmf were not found."
        )

    list_nucleation_size = compute_critical_nucleation(
        fl33_file,
        "yaml_files/material.yaml",
        "yaml_files/fault_slip.yaml",
        list_fault_yaml,
        hypo,
    )
    print(list_nucleation_size)

    for k, (idx, row) in enumerate(param_df.iterrows()):
        if list_nucleation_size[k]:
            template_par, code = extract_template_params(
                idx,
                row,
                Cname,
                cohesion_values,
                hypo,
                max_slip,
                constant_d_c,
                input_config,
                derived_config,
                CFS_code_placeholder,
            )
            template_par["r_crit"] = list_nucleation_size[k]
            fn_fault = f"yaml_files/fault_{code}.yaml"
            render_file(templateEnv, template_par, "fault.tmpl.yaml", fn_fault)
        else:
            fn_param = f"parameters_dyn_{code}.par"
            print(f"removing {fn} and {fn_param} (nucleation too large)")
            os.remove(fn)
            os.remove(fn_param)

    # write parts.txt files
    parameter_files = sorted(
        [f for f in os.listdir(".") if f.startswith("parameters_dyn")]
    )
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
