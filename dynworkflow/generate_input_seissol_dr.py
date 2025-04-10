#!/usr/bin/env python3
import numpy as np
import glob
import re
import os
import jinja2
from sklearn.decomposition import PCA
import shutil
from dynworkflow.estimate_nucleation_radius import compute_critical_nucleation
from scipy.stats import qmc
import random
import itertools
from dynworkflow.compile_scenario_macro_properties import infer_duration
import warnings
import seissolxdmf as sx
import argparse
from pyproj import Transformer


def generate(mode, dic_values):
    if not os.path.exists("yaml_files"):
        os.makedirs("yaml_files")

    def compute_max_slip(fn):
        sx0 = sx.seissolxdmf(fn)
        ndt = sx0.ReadNdt()
        ASl = sx0.ReadData("ASl", ndt - 1)
        if np.any(np.isnan(ASl)):
            ASl = sx0.ReadData("ASl", ndt - 2)
        return ASl.max()

    kinmod_fn = "output/dyn-kinmod-fault.xdmf"
    if not os.path.exists(kinmod_fn):
        kinmod_fn = "extracted_output/dyn-kinmod_extracted-fault.xdmf"
    max_slip = compute_max_slip(kinmod_fn)
    assert max_slip > 0

    # Get the directory of the script
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    input_file_dir = f"{script_directory}/input_files"
    templateLoader = jinja2.FileSystemLoader(searchpath=input_file_dir)
    templateEnv = jinja2.Environment(loader=templateLoader)
    number_of_segments = len(glob.glob("tmp/*.ts"))
    print(f"found {number_of_segments} segments")

    assert mode in ["latin_hypercube", "grid_search", "picked_models"]
    longer_and_more_frequent_output = mode == "picked_models"

    if mode == "latin_hypercube":

        def get_min_max(dic_values, key):
            values = dic_values[key]
            return min(values), max(values)

        R0, R1 = get_min_max(dic_values, "R")
        B0, B1 = get_min_max(dic_values, "B")
        C0, C1 = get_min_max(dic_values, "C")

        l_bounds = [R0, B0, C0]
        u_bounds = [R1, B1, C1]

        # cohesion is fixed in this appraoch
        list_cohesion = dic_values["cohesion"]
        assert len(list_cohesion) == 1

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
        nsample = dic_values["nsamples"]
        sample = sampler.random(n=nsample)
        pars = qmc.scale(sample, l_bounds, u_bounds)
        pars = np.around(pars, decimals=3)
        column_of_zeros = np.zeros((1, nsample))
        pars = np.insert(pars, 0, column_of_zeros, axis=1)

    elif mode == "grid_search":
        # grid parameter space
        paramB = dic_values["B"]
        paramC = dic_values["C"]
        paramR = dic_values["R"]
        list_cohesion = dic_values["cohesion"]
        paramCoh = list(range(len(list_cohesion)))
        use_R_segment_wise = False
        if use_R_segment_wise:
            params = [paramCoh, paramB, paramC] + [paramR] * number_of_segments
            assert len(params) == number_of_segments + 3
        else:
            params = [paramCoh, paramB, paramC, paramR]
            assert len(params) == 4
        # Generate all combinations of parameter values
        param_combinations = list(itertools.product(*params))
        # Convert combinations to numpy array and round to desired decimals
        pars = np.around(np.array(param_combinations), decimals=3)
        print(pars)
    elif mode == "picked_models":
        list_cohesion = dic_values["cohesion"]
        n = len(list_cohesion)
        assert len(dic_values["B"]) == len(dic_values["C"]) == len(dic_values["R"]) == n
        pars = [
            [i, dic_values["B"][i], dic_values["C"][i], dic_values["R"][i]]
            for i in range(n)
        ]
        pars = np.array(pars)
    else:
        raise NotImplementedError(f"unkown mode {mode}")

    nsample = pars.shape[0]
    print(f"parameter space has {nsample} samples")

    def render_file(template_par, template_fname, out_fname, verbose=True):
        template = templateEnv.get_template(template_fname)
        outputText = template.render(template_par)
        with open(out_fname, "w") as fid:
            fid.write(outputText)
        if verbose:
            print(f"done creating {out_fname}")

    with open("tmp/projection.txt", "r") as f:
        projection = f.read()
    transformer = Transformer.from_crs("epsg:4326", projection, always_xy=True)
    hypo = np.loadtxt("tmp/hypocenter.txt")
    hypo[2] *= -1e3
    hypo[0], hypo[1] = transformer.transform(hypo[0], hypo[1])

    fn_mr = "tmp/moment_rate_from_finite_source_file.txt"
    moment_rate = np.loadtxt(fn_mr)
    kinmod_duration = infer_duration(moment_rate[:, 0], moment_rate[:, 1])

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

    fault_sampling = compute_fault_sampling(kinmod_duration)

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

    list_fault_yaml = []
    for i in range(nsample):
        row = pars[i, :]
        cohi, B, C = row[0:3]
        cohesion_const, cohesion_lin = list_cohesion[int(cohi)]
        R = row[3:]

        template_par = {
            "R_yaml_block": generate_R_yaml_block(R),
            "cohesion_const": cohesion_const * 1e6,
            "cohesion_lin": cohesion_lin * 1e6,
            "B": B,
            "C": C,
            "min_dc": C * max_slip * 0.15,
            "hypo_x": hypo[0],
            "hypo_y": hypo[1],
            "hypo_z": hypo[2],
            "r_crit": 3000.0,
        }

        sR = "_".join(map(str, R))
        code = f"{i:04}_coh{cohesion_const}_{cohesion_lin}_B{B}_C{C}_R{sR}"
        fn_fault = f"yaml_files/fault_{code}.yaml"
        list_fault_yaml.append(fn_fault)

        render_file(template_par, "fault.tmpl.yaml", fn_fault)

        if longer_and_more_frequent_output:
            template_par["terminatorMomentRateThreshold"] = -1
            template_par["surface_output_interval"] = 1.0
        else:
            template_par["terminatorMomentRateThreshold"] = 5e17
            template_par["surface_output_interval"] = 5.0
        template_par["end_time"] = kinmod_duration + max(20.0, 0.25 * kinmod_duration)
        template_par["fault_fname"] = fn_fault
        template_par["output_file"] = f"output/dyn_{code}"
        template_par["material_fname"] = "yaml_files/material.yaml"
        template_par["fault_print_time_interval"] = fault_sampling
        fn_param = f"parameters_dyn_{code}.par"
        render_file(template_par, "parameters_dyn.tmpl.par", fn_param)

    fnames = ["smooth_PREM_material.yaml", "mud.yaml", "fault_slip.yaml"]
    for fn in fnames:
        shutil.copy(f"{input_file_dir}/{fn}", f"yaml_files/{fn}")

    if os.path.exists("output/fl33-fault.xdmf"):
        fl33_file = "output/fl33-fault.xdmf"
    elif os.path.exists("extracted_output/fl33_extracted-fault.xdmf"):
        fl33_file = "extracted_output/fl33_extracted-fault.xdmf"
    else:
        raise FileNotFoundError(
            "The files output/fl33-fault.xdmf or extracted_output/fl33_extracted-fault.xdmf were not found."
        )

    list_nucleation_size = compute_critical_nucleation(
        fl33_file,
        "yaml_files/smooth_PREM_material.yaml",
        "yaml_files/fault_slip.yaml",
        list_fault_yaml,
        hypo,
    )
    print(list_nucleation_size)
    for i, fn in enumerate(list_fault_yaml):
        row = pars[i, :]
        cohi, B, C = row[0:3]
        cohesion_const, cohesion_lin = list_cohesion[int(cohi)]
        R = row[3:]
        sR = "_".join(map(str, R))
        code = f"{i:04}_coh{cohesion_const}_{cohesion_lin}_B{B}_C{C}_R{sR}"
        if list_nucleation_size[i]:
            fn_fault = f"yaml_files/fault_{code}.yaml"
            assert fn_fault == fn
            template_par = {
                "R_yaml_block": generate_R_yaml_block(R),
                "cohesion_const": cohesion_const * 1e6,
                "cohesion_lin": cohesion_lin * 1e6,
                "B": B,
                "C": C,
                "min_dc": C * max_slip * 0.15,
                "hypo_x": hypo[0],
                "hypo_y": hypo[1],
                "hypo_z": hypo[2],
                "r_crit": list_nucleation_size[i],
            }
            render_file(template_par, "fault.tmpl.yaml", fn_fault)
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


if __name__ == "__main__":
    # Default values for parameters
    paramB = [0.9, 1.0, 1.1, 1.2]
    # paramC = [0.1, 0.15, 0.2, 0.25, 0.3]
    paramC = [0.1, 0.2, 0.3, 0.4, 0.5]
    paramR = [0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
    paramCoh = [(0.25, 1)]

    def list_to_semicolon_separated_string(li):
        return ";".join(str(v) for v in li)

    def semicolon_separated_string_to_list(li):
        return [float(v) for v in li.split(";")]

    def list_of_tuples_to_semicolon_separated_string(li):
        return ";".join(f"{v[0]} {v[1]}" for v in li)

    def semicolon_separated_string_to_list_of_tuples(s):
        return [tuple(map(float, w.split())) for w in s.split(";")]

    parser = argparse.ArgumentParser(
        description="automatically generate input files for dynamic rupture models"
    )
    parser.add_argument(
        "mode",
        default="grid_search",
        choices=["latin_hypercube", "grid_search", "picked_models"],
        help="mode use to sample the parameter space",
    )
    parser.add_argument(
        "--Bvalues",
        nargs=1,
        help="B (stress drop factor) values, separated by ';'",
        default=list_to_semicolon_separated_string(paramR),
    )
    parser.add_argument(
        "--Cvalues",
        nargs=1,
        help="C (Dc slip factor) values, separated by ';'",
        default=list_to_semicolon_separated_string(paramR),
    )
    parser.add_argument(
        "--Rvalues",
        nargs=1,
        help="R (relative prestress) values, separated by ';'",
        default=list_to_semicolon_separated_string(paramR),
    )
    parser.add_argument(
        "--cohesionvalues",
        nargs=1,
        help="fault cohesion (c0 + c1*sigma_zz) values, 2 value per parameter set, separated by';'",
        default=list_of_tuples_to_semicolon_separated_string(paramCoh),
    )
    parser.add_argument(
        "--nsamples",
        nargs=1,
        help="number of samples (for Latin hypercube)",
        default=[50],
    )

    args = parser.parse_args()
    dic_values = {}
    dic_values["B"] = semicolon_separated_string_to_list(args.Bvalues[0])
    dic_values["C"] = semicolon_separated_string_to_list(args.Cvalues[0])
    dic_values["R"] = semicolon_separated_string_to_list(args.Rvalues[0])
    dic_values["cohesion"] = semicolon_separated_string_to_list_of_tuples(
        args.cohesionvalues[0]
    )
    dic_values["nsamples"] = args.nsamples[0]
    print(dic_values)
    generate(args.mode, dic_values)
