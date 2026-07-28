#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import glob
import os
import pickle
import re

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import yaml
from obspy.signal.cross_correlation import correlate, xcorr_max
from scipy import integrate

from dynworkflow import step1_args
from dynworkflow.plot_utils import plot_combined_gof_plot, plot_moment_rates
from dynworkflow.stf_loader import read_usgs_moment_rate, trim_trailing_zero

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


def parse_parameter_string(param_str):
    # todo: remove duplicated function
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


def infer_duration(time, moment_rate):
    moment = integrate.cumulative_trapezoid(moment_rate, time, initial=0)
    M0 = np.trapezoid(moment_rate, x=time)
    return np.amax(time[moment < 0.99 * M0])


def computeMw(label, time, moment_rate):
    M0 = np.trapezoid(moment_rate, x=time)
    Mw = 2.0 * np.log10(M0) / 3.0 - 6.07
    # print(f"{label} moment magnitude: {Mw:.2} (M0 = {M0:.4e})")
    return M0, Mw


def plot_gof_xy(df, gof1, gof2):
    if gof1 in df.keys() and gof2 in df.keys():
        print(f"skipping plot_gof_xy with {gof1} {gof2} as not found in {df.keys()}")
        return

    # Plot
    plt.figure(figsize=(8, 6))
    print(df.keys())
    plt.scatter(df[gof1], df[gof2], alpha=0.7)
    plt.xlabel(gof1)
    plt.ylabel(gof2)
    plt.grid(True)
    plt.savefig(f"plots/gof_plot_{gof1}_{gof2}.png", dpi=300, bbox_inches="tight")
    plt.close()


def extract_params_from_prefix(fname: str) -> dict:
    """
    Extract simulation parameters from a file name prefix.

    Parameters:
    fname (str): File name to extract parameters from.

    Returns:
    dict: A dictionary containing the extracted parameters:
        - sim_id (int): Simulation ID
        - coh (tuple[float, float] or float): Cohesion values (or NaN if not present)
        - B (float): B value
        - C (float): C value
        - R (float or list[float]): R value(s)

    Raises:
    ValueError: If no match is found in the file name prefix.
    """

    def extract_BCR(out: dict, match, i0: int = 1) -> None:
        """
        Extract B, C, and R values from a match object.

        Parameters:
        out (dict): Dictionary to store the extracted values.
        match: Match object from re.search.
        i0 (int): Starting index for group extraction.
        """
        out["B"] = float(match.group(i0))
        out["C"] = float(match.group(i0 + 1))
        R_value = list(map(float, match.group(i0 + 2).split("_")))
        if len(R_value) == 1:
            R_value = R_value[0]
        out["R"] = R_value

    patterns = [
        r"dyn[/_-]([^_]+)_coh([\d.]+)_([\d.]+)_B([\d.]+)_C([\d.]+)_R([\d._]+)"
        r"(?:_[^_]+)?-energy.csv",
        r"dyn[/_-]([^_]+)_B([\d.]+)_C([\d.]+)_R([\d._]+)(?:_[^_]+)?energy.csv",
    ]

    for i in range(2):
        match = re.search(patterns[i], fname)
        out = {}
        if i == 0 and match:
            out["sim_id"] = int(match.group(1))
            out["coh"] = (float(match.group(2)), float(match.group(3)))
            extract_BCR(out, match, 4)
            break
        elif i == 1 and match:
            out["sim_id"] = int(match.group(1))
            extract_BCR(out, match, 2)
            out["coh"] = np.nan
    if not out:
        out = {"coh": np.nan, "sim_id": -1, "B": np.nan, "C": np.nan, "R": np.nan}
    return out


def main(args):
    args.nmin = min(args.nmin, args.nmax)

    ps = args.font_size[0]
    matplotlib.rcParams.update({"font.size": ps})
    plt.rcParams["font.family"] = "sans"
    matplotlib.rc("xtick", labelsize=ps)
    matplotlib.rc("ytick", labelsize=ps)

    if not os.path.exists("plots"):
        os.makedirs("plots")

    if args.output_folder.endswith("energy.csv"):
        energy_files = [args.output_folder]
    else:
        if os.path.exists(args.output_folder):
            args.output_folder += "/"
        energy_files = sorted(glob.glob(f"{args.output_folder}*-energy.csv"))

    # remove fl33
    energy_files = [s for s in energy_files if "fl33" not in s]

    fn = "tmp/reference_STF.txt"

    # load first default arguments for backwards compatibility
    input_config_dict = step1_args.get_args()

    with open("input_config.yaml", "r") as f:
        input_config_dict |= yaml.safe_load(f)

    parameters_structured = parse_parameter_string(input_config_dict["parameters"])
    parameter_names = list(parameters_structured.keys())
    parameter_names = [name for name in parameter_names if name != "cohesion"]
    parameter_names_with_coh = ["coh"] + parameter_names

    def unpack_gof_components_and_weights(gof_components_descr):
        gof_components = gof_components_descr.strip().split(",")
        gof_weights = {}
        for i, comp_and_weight in enumerate(gof_components):
            parts = comp_and_weight.split()
            if len(parts) > 2:
                raise ValueError("did not understand format of gof_component: {comp}")
            elif len(parts) == 2:
                comp = parts[0]
                gof_weights[comp] = float(parts[1])
            elif len(parts) == 1:
                gof_weights[comp_and_weight] = 1.0
        # Normalize the weights so they sum to 1
        total_weight = sum(gof_weights.values())
        gof_weights = {k: v / total_weight for k, v in gof_weights.items()}
        return gof_weights

    gof_weights = unpack_gof_components_and_weights(input_config_dict["gof_components"])
    gof_components = gof_weights.keys()
    print(gof_weights)

    gof_component_to_name = {}
    gof_component_to_name["slip_distribution"] = "gof_slip"
    gof_component_to_name["teleseismic_body_wf"] = "gof_body_wf"
    gof_component_to_name["teleseismic_surface_wf"] = "gof_surf_wf"
    gof_component_to_name["regional_wf"] = "gof_reg"
    gof_component_to_name["moment_rate_function"] = "gof_MRF"
    gof_component_to_name["fault_offsets"] = "gof_offsets"
    gof_component_to_name["slip_rate"] = "gof_slip_rate"
    gof_component_to_name["seismic_moment"] = "gof_M0"

    if os.path.exists("derived_config.yaml"):
        with open("derived_config.yaml", "r") as f:
            config_dict = yaml.safe_load(f)

        ref_stfs = config_dict.get("reference_STFs", None)
        if not ref_stfs:
            raise ValueError(
                "reference_STFs not found in derived_config.yaml. "
                "If using an old setup, change the reference_STF entry to a "
                "reference_STFs: [[file_name, label]] format."
            )
        else:
            # Take the first reference STF for ranking
            first_ref = ref_stfs[0]
            refMRFfile, ref_name = first_ref

    elif os.path.exists(fn):
        ref_stfs = []
        # for backwards compatibility
        with open("tmp/reference_STF.txt", "r") as fid:
            refMRFfile = fid.read().strip()
    else:
        ref_stfs = []
        # for backwards compatibility
        print(f"{fn} does not exists!")
        if os.path.exists("tmp/moment_rate_from_finite_source_file_usgs.txt"):
            refMRFfile = "tmp/moment_rate.mr"
        else:
            refMRFfile = "tmp/moment_rate_from_finite_source_file.txt"

    ref_name = "finite-source model"
    if refMRFfile == "tmp/moment_rate_from_finite_source_file.txt":
        mr_ref = np.loadtxt("tmp/moment_rate_from_finite_source_file.txt")
    else:
        mr_ref = read_usgs_moment_rate(refMRFfile)

    mr_ref = trim_trailing_zero(mr_ref)

    M0ref, Mwref = computeMw(ref_name, mr_ref[:, 0], mr_ref[:, 1])

    inferred_duration = infer_duration(mr_ref[:, 0], mr_ref[:, 1])

    # Create a new time array with the desired time step
    dt = 0.25
    new_time = np.arange(mr_ref[:, 0].min(), mr_ref[:, 0].max() + dt, dt)
    # Use numpy's interp function to interpolate values at the new time points
    mr_ref_interp = np.interp(new_time, mr_ref[:, 0], mr_ref[:, 1])

    if 2 * inferred_duration > new_time[-1]:
        added = int((2 * inferred_duration - new_time.max()) / dt)
        mr_ref_interp = np.pad(
            mr_ref_interp, (0, added), "constant", constant_values=(0, 0)
        )

    param_files = sorted(glob.glob("simulation_parameter*.csv"))

    # Initialize params and Cname
    if param_files:
        # Read and concatenate all parameter files
        df_list = [pd.read_csv(f) for f in param_files]
        params = pd.concat(df_list, ignore_index=True)
        params["sim_id"] = params["id"].astype(int)
        Cname = "dc" if "dc" in params.columns else "C"
    else:
        params = None
        Cname = "C"

    results = {
        "coh": [],
        "sim_id": [],
        "Mw": [],
        "duration": [],
        "gof_MRF": [],
        "gof_M0": [],
        "gof_T": [],
        "faultfn": [],
    }
    for name in parameter_names:
        results[name] = []

    for i, fn in enumerate(energy_files):
        df = pd.read_csv(fn)
        df = df.pivot_table(index="time", columns="variable", values="measurement")
        if len(df) < 2 or df["seismic_moment"].iloc[-1] == 0.0:
            print(f"skipping empty or 0 seismic moment: {fn}")
            continue
        dt = df.index[1] - df.index[0]
        assert dt == 0.25
        df["seismic_moment_rate"] = np.gradient(df["seismic_moment"], dt)
        label = os.path.basename(fn)
        prefix = fn.split("-energy.csv")[0]
        faultfn = glob.glob(f"{prefix}-fault.xdmf")
        if faultfn:
            faultfn = faultfn[0]
        else:
            faultfn = glob.glob(f"{prefix}_*-fault.xdmf")[0]
        if params is None:
            out = extract_params_from_prefix(fn)
            coh_name = "coh"
        else:
            match = re.search(r"dyn_(\d{4})_", os.path.basename(fn))
            if match:
                sim_id = int(match.group(1))
            else:
                raise ValueError(f"could not get sim_id from {fn}")
            out = params[params["id"] == sim_id]
            out = out.to_dict(orient="records")[0]
            coh_name = "cohesion_value"

        M0, Mw = computeMw(label, df.index.values, df["seismic_moment_rate"])
        duration = infer_duration(df.index.values, df["seismic_moment_rate"])
        results["Mw"].append(Mw)
        results["duration"].append(duration)
        results["sim_id"].append(int(out["sim_id"]))
        results["coh"].append(out[coh_name])
        for name in parameter_names:
            results[name].append(out[name])

        results["faultfn"].append(faultfn)
        # max_shift = int(min(5, 0.25 * inferred_duration) / dt)
        max_shift = 0

        len_corr = max(len(mr_ref_interp), len(df["seismic_moment_rate"]))
        # signal padded for easier interpretation of the shift
        s1 = np.pad(
            mr_ref_interp,
            (0, len_corr - len(mr_ref_interp)),
            "constant",
            constant_values=(0, 0),
        )
        s2 = np.pad(
            df["seismic_moment_rate"],
            (0, len_corr - len(df["seismic_moment_rate"])),
            "constant",
            constant_values=(0, 0),
        )
        cc = correlate(s1, s2, shift=max_shift)
        shift, ccmax = xcorr_max(cc, abs_max=False)
        # results["shift_syn_ref_sec"].append(shift * dt)
        results["gof_MRF"].append(ccmax)
        # allow 15% variation on the misfit
        M0_gof = min(1, 1.15 - abs(M0 - M0ref) / M0ref)
        results["gof_M0"].append(M0_gof)
        duration_gof = min(1, 1 - abs(duration - inferred_duration) / inferred_duration)
        results["gof_T"].append(duration_gof)
    result_df = pd.DataFrame(results)
    result_df["Mw"] = result_df["Mw"].round(3)

    def load_and_merge(df, filename, id_col, value_cols):
        regex = r"dyn[/_-]([^_]+)_"
        if not os.path.exists(filename):
            print(f"{filename} could not be found")
            return df

        # Load based on extension
        data = (
            pickle.load(open(filename, "rb"))
            if filename.endswith(".pkl")
            else pd.read_csv(filename)
        )

        # Extract sim_id safely
        extracted = data[id_col].str.extract(regex)[0]
        data["sim_id"] = pd.to_numeric(extracted, errors="coerce")
        data = data.dropna(subset=["sim_id"]).copy()
        data["sim_id"] = data["sim_id"].astype(int)

        # Select only necessary columns and merge
        return pd.merge(df, data[["sim_id"] + value_cols], on="sim_id", how="left")

    result_df = load_and_merge(result_df, "gof_slip.pkl", "faultfn", ["gof_slip"])
    result_df = load_and_merge(
        result_df, "percentage_supershear.pkl", "faultfn", ["supershear"]
    )
    result_df = load_and_merge(result_df, "rms_offset.csv", "faultfn", ["offset_rms"])
    result_df = load_and_merge(
        result_df, "rms_slip_rate.csv", "fault_receiver_fname", ["slip_rate_rms"]
    )
    result_df = load_and_merge(
        result_df,
        "area_max_R.csv",
        "faultfn",
        ["area_max_R"],
    )
    result_df = load_and_merge(result_df, "Gc.csv", "faultfn", ["Gc"])

    if "offset_rms" in result_df.columns:
        result_df["gof_offsets"] = np.exp(-result_df["offset_rms"])

    if "slip_rate_rms" in result_df.columns:
        result_df["gof_slip_rate"] = np.exp(-result_df["slip_rate_rms"])

    if "area_max_R" in result_df.columns:
        result_df["area_max_R"] = result_df["area_max_R"].round(1)
        # Be careful: this drops rows from the entire result_df
        result_df = result_df[result_df["area_max_R"] < 1000.0]

    result_df = result_df.drop(
        columns=["offset_rms", "slip_rate_rms", "gof_M0", "gof_T"], errors="ignore"
    )

    def compute_weighted_wf_gof(gof_df, gof_wf_weights, gof_name):
        df_all = None
        available_names = gof_df["gofa_name"].unique()
        for pattern, weight in gof_wf_weights.items():
            # Filter for the pattern
            df_pattern = gof_df[gof_df["gofa_name"].str.contains(rf"{pattern}\d+")]
            if df_pattern.empty:
                raise ValueError(
                    f"No rows found for pattern '{pattern}\\d+' while"
                    f"computing '{gof_name}'.\n"
                    f"Available gofa_name values:\n{available_names}"
                )

            # Rename 'gofa' column to keep it separate
            df_pattern = df_pattern.rename(columns={"gofa": f"gofa_{pattern}"})

            # Merge on 'sim_id'
            if df_all is None:
                df_all = df_pattern
            else:
                df_all = pd.merge(df_all, df_pattern, on="sim_id", how="outer")

        # Compute weighted sum
        weighted_cols = [f"gofa_{pattern}" for pattern in gof_wf_weights]
        df_all[gof_name] = sum(
            df_all[col] * gof_wf_weights[pattern]
            for col, pattern in zip(weighted_cols, gof_wf_weights)
        )
        return df_all[[gof_name, "sim_id"]]

    # read weights
    gof_weights_reg = unpack_gof_components_and_weights(
        input_config_dict["regional_wf_components"]
    )
    gof_weights_body = unpack_gof_components_and_weights(
        input_config_dict["teleseismic_body_wf_components"]
    )
    gof_weights_surf = unpack_gof_components_and_weights(
        input_config_dict["teleseismic_surface_wf_components"]
    )

    for waveform_type in ["regional", "global"]:
        pattern = f"gof_*_{waveform_type}_waveforms_average.pkl"
        matching_files = glob.glob(pattern)
        if len(matching_files) > 0:
            fname = matching_files[0]
            print(f"{fname} detected: merging with results dataframe")
            gofa = pickle.load(open(fname, "rb"))
            gofa = gofa[~gofa["src"].apply(lambda x: "kinmod" in x[1])]

            if waveform_type == "regional":
                processed_gof = compute_weighted_wf_gof(
                    gofa, gof_weights_reg, "gof_reg"
                )
                result_df = pd.merge(result_df, processed_gof, on="sim_id", how="left")
            else:
                processed_gof = compute_weighted_wf_gof(
                    gofa, gof_weights_body, "gof_body_wf"
                )
                result_df = pd.merge(result_df, processed_gof, on="sim_id", how="left")
                processed_gof = compute_weighted_wf_gof(
                    gofa, gof_weights_surf, "gof_surf_wf"
                )
                result_df = pd.merge(result_df, processed_gof, on="sim_id", how="left")

    if "gof_body_wf" in result_df.keys():
        plot_gof_xy(result_df, "gof_body_wf", "gof_MRF")
        plot_gof_xy(result_df, "gof_body_wf", "duration")

    result_df = result_df[sorted(result_df.columns)]
    result_df["combined_gof"] = 0.0
    sum_weights = 0
    component_used = {}
    for comp in gof_components:
        col_name = gof_component_to_name[comp]
        if col_name in result_df.keys():
            result_df["combined_gof"] += result_df[col_name] * gof_weights[comp]
            sum_weights += gof_weights[comp]
            component_used[comp] = gof_weights[comp]
        else:
            Warning(
                f"{comp} given in 'gof_components' ({gof_components})"
                f"but {col_name} not found in result_df"
            )
    if sum_weights != 0.0:
        result_df["combined_gof"] /= sum_weights
        component_used = {k: v / sum_weights for k, v in component_used.items()}

    result_df = result_df.sort_values(by="combined_gof", ascending=False).reset_index(
        drop=True
    )

    result_df["faultfn"] = result_df["faultfn"].apply(
        lambda x: os.path.basename(x)
        .split("_extracted-fault.xdmf")[0]
        .split("_compacted-fault.xdmf")[0]
        .split("-fault.xdmf")[0]
    )
    result_df.to_pickle("compiled_results.pkl")
    print(result_df.to_string())

    varying_param = {}
    for name in parameter_names_with_coh:
        varying_param[name] = len(np.unique(result_df[name].values)) > 1

    plot_moment_rates(
        energy_files=energy_files,
        result_df=result_df,
        args=args,
        ref_stfs=ref_stfs,
        mr_ref=mr_ref,
        ref_name=ref_name,
        Mwref=Mwref,
        varying_param=varying_param,
    )

    selected_rows = result_df[result_df["Mw"] > 6]
    selected_rows = selected_rows.sort_values(
        by="combined_gof", ascending=False
    ).reset_index(drop=True)
    print(selected_rows.to_string())

    if "combined_gof" in selected_rows.columns:
        selected_rows = selected_rows.sort_values(
            by="combined_gof", ascending=False
        ).reset_index(drop=True)
        print(selected_rows.to_string())

    fname = "tmp/selected_output.txt"
    with open(fname, "w") as fid:
        for index, row in selected_rows.iterrows():
            fid.write(f"{row['faultfn']}\n")
            if index == 3:
                break
        fid.write("output/dyn-usgs-fault.xdmf\n")
    print(f"done writing {fname}")

    print(f"components used for combined_gof: {component_used}")
    if "regional_wf" in component_used.keys():
        print(f"components used for regional_wf: {gof_weights_reg}")
    if "teleseismic_body_wf" in component_used.keys():
        print(f"components used for teleseismic_body_wf: {gof_weights_body}")
    if "teleseismic_surface_wf" in component_used.keys():
        print(f"components used for teleseismic_surface_wf: {gof_weights_surf}")

    if not result_df.empty:
        keys_to_plot = [key for key in result_df.keys() if "gof" in key]
        nlines = len(keys_to_plot) // 3
        preferred_model = {
            "B": result_df["B"].iloc[0],
            Cname: result_df[Cname].iloc[0],
            "R": result_df["R"].iloc[0],
        }
        preferred_model = {k: float(v) for k, v in preferred_model.items()}
        plot_combined_gof_plot(result_df, keys_to_plot, nlines, preferred_model)
