#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import yaml
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import argparse
import matplotlib
import os
import glob
import re
from obspy.signal.cross_correlation import correlate, xcorr_max
from scipy import integrate
from cmcrameri import cm
import pickle

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
    M0 = np.trapz(moment_rate[:], x=time[:])
    return np.amax(time[moment < 0.99 * M0])


def computeMw(label, time, moment_rate):
    M0 = np.trapz(moment_rate[:], x=time[:])
    Mw = 2.0 * np.log10(M0) / 3.0 - 6.07
    # print(f"{label} moment magnitude: {Mw:.2} (M0 = {M0:.4e})")
    return M0, Mw


def plot_gof_xy(df, gof1, gof2):
    if gof1 in df.keys() and gof2 in df.keys():
        print("skipping plot_gof_xy with {gof1} {gof2} as not found in {df.keys()}")
        return

    # Plot
    plt.figure(figsize=(8, 6))
    print(result_df.keys())
    plt.scatter(result_df[gof1], result_df[gof2], alpha=0.7)
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


def generate_XY_panel(
    name_arr1, arr1, name_arr2, arr2, name3, val3, name_col, ax, cmap
):
    "generate a 2D plot with name_arr1 and name_arr2 in X and Y axis"
    "colored by name_col"
    unique_arr1 = np.unique(arr1)
    unique_arr2 = np.unique(arr2)
    X, Y = np.meshgrid(unique_arr1, unique_arr2)
    gof_array = np.zeros_like(X) + np.nan
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            result = result_df[
                (result_df[name_arr1] == X[i, j])
                & (result_df[name_arr2] == Y[i, j])
                & (result_df[name3] == val3)
            ]
            if not result.empty:
                specific_index = result.index[0]
                gof_array[i, j] = result.loc[specific_index, name_col]
    im = ax.pcolormesh(X, Y, gof_array, cmap=cmap)

    mask_invalid = np.ma.masked_where(np.isfinite(gof_array), gof_array)
    mask_invalid[np.isnan(gof_array)] = 1
    ax.pcolor(X, Y, mask_invalid, hatch="/", alpha=0)
    if name_col in ["Mw"]:
        im.set_clim(result_df[name_col].min(), result_df[name_col].max())
    # else:
    #    im.set_clim(0, result_df[name_col].max())

    ax.set_xlabel(name_arr1)
    ax.set_ylabel(name_arr2)
    ax.set_xticks(unique_arr1)
    from matplotlib.ticker import FormatStrFormatter

    ax.set_xticklabels(FormatStrFormatter("%g").format_ticks(unique_arr1))
    ax.set_yticks(unique_arr2)
    ax.set_title(f"{name3}={val3}")
    if name_col == "gof_MRF":
        label = "gof moment rate function"
    elif name_col == "gof_M0":
        label = "gof seismic moment"
    elif name_col == "gof_tel_wf":
        label = "gof teleseismic waveforms"
    elif name_col == "gof_reg":
        label = "gof regional waveforms"
    elif name_col == "combined_gof":
        label = "combined gof"
    else:
        label = name_col

    plt.colorbar(im, label=label, ax=ax)


def generate_BCR_plots(B, C, R):
    unique_R = np.unique(R)
    n_div = 2
    nrow, ncol = int(np.ceil(len(unique_R) / n_div)), 2 * n_div
    # nrow, ncol = 2, 2
    fig, axarr = plt.subplots(
        nrow,
        ncol,
        figsize=(ncol * 4, nrow * 4),
        dpi=160,
        sharex=False,
        sharey=True,
        squeeze=False,
    )
    for k, Rk in enumerate(unique_R):
        row = k % nrow
        col = k // nrow * n_div
        generate_XY_panel(
            "B", B, Cname, C, "R0", Rk, "gof_MRF", axarr[row, col], cm.cmaps["acton_r"]
        )
        generate_XY_panel(
            "B",
            B,
            Cname,
            C,
            "R0",
            Rk,
            "gof_M0",
            axarr[row, col + 1],
            cm.cmaps["oslo_r"],
        )
    fname = "plots/parameter_space_BC_constant_R.pdf"
    plt.savefig(fname)
    print(f"done writing {fname}")

    unique_B = np.unique(B)
    n_div = 1
    nrow, ncol = len(unique_B) // n_div, 3 * n_div
    fig, axarr = plt.subplots(
        nrow,
        ncol,
        figsize=(ncol * 4, nrow * 4),
        dpi=160,
        sharex=False,
        sharey=True,
        squeeze=False,
    )

    for k, Bk in enumerate(unique_B):
        row = k % nrow
        col = k // nrow * n_div
        generate_XY_panel(
            "R0",
            R,
            Cname,
            C,
            "B",
            Bk,
            "combined_gof",
            axarr[row, col],
            cm.cmaps["acton_r"],
        )
        if "gof_reg" in result_df:
            generate_XY_panel(
                "R0",
                R,
                Cname,
                C,
                "B",
                Bk,
                "gof_reg",
                axarr[row, col + 1],
                cm.cmaps["acton_r"],
            )
            generate_XY_panel(
                "R0",
                R,
                Cname,
                C,
                "B",
                Bk,
                # "combined_gof",
                "combined_gof",
                axarr[row, col + 2],
                cm.cmaps["batlowW_r"],
            )

    fname = "plots/parameter_space_R0C_constant_B.pdf"
    plt.savefig(fname)
    print(f"done writing {fname}")


def generate_BCR_moment_plots(B, C, R):
    unique_B = np.unique(B)
    n_div = 1
    nrow, ncol = len(unique_B) // n_div, 2 * n_div
    fig, axarr = plt.subplots(
        nrow,
        ncol,
        figsize=(ncol * 4, nrow * 4),
        dpi=160,
        sharex=False,
        sharey=True,
        squeeze=False,
    )

    for k, Bk in enumerate(unique_B):
        row = k % nrow
        col = k // nrow * n_div
        generate_XY_panel(
            "R0",
            R,
            Cname,
            C,
            "B",
            Bk,
            "Mw",
            axarr[row, col],
            cm.cmaps["acton"],
        )
        generate_XY_panel(
            "R0",
            R,
            Cname,
            C,
            "B",
            Bk,
            "duration",
            axarr[row, col + 1],
            cm.cmaps["oslo"],
        )

    fname = "plots/parameter_space_Mw_duration_R0C_constant_B.pdf"
    plt.savefig(fname)
    print(f"done writing {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute scenario properties")
    parser.add_argument(
        "output_folder",
        help="path to output folder or full path to specific energy.csv file",
    )
    parser.add_argument(
        "--extension",
        help="figure extension (without the .)",
        default="pdf",
    )
    parser.add_argument("--font_size", help="font size", nargs=1, default=[8], type=int)

    parser.add_argument(
        "--gof_threshold",
        help="gof threshold from which results are selected",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--nmin",
        help="minimum number of synthetic moment rates drawn in color",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--nmax",
        help="maximum number of synthetic moment rates drawn in color",
        type=int,
        default=10,
    )

    args = parser.parse_args()

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
    one_model_shown = args.nmax == 1
    matplotlib.rcParams["lines.linewidth"] = 0.5 if one_model_shown else 1.0

    centimeter = 1 / 2.54
    figsize = (7.5 * centimeter, 4.0 * centimeter) if one_model_shown else (8, 4)
    fig = plt.figure(figsize=figsize, dpi=80)
    ax = fig.add_subplot(111)

    def read_usgs_moment_rate(fname):
        mr_ref = np.loadtxt(fname, skiprows=2)
        # Conversion factor from dyne-cm/sec to Nm/sec (for older usgs files)
        scaling_factor = 1.0 if np.amax(mr_ref[:, 1]) < 1e23 else 1e-7
        mr_ref[:, 1] *= scaling_factor
        return mr_ref

    def trim_trailing_zero(mr_ref):
        last_index_non_zero = np.nonzero(mr_ref[:, 1])[0][-1]
        return mr_ref[:last_index_non_zero, :]

    fn = "tmp/reference_STF.txt"

    with open("input_config.yaml", "r") as f:
        input_config_dict = yaml.safe_load(f)

    parameters_structured = parse_parameter_string(input_config_dict["parameters"])
    parameter_names = list(parameters_structured.keys())
    parameter_names = [name for name in parameter_names if name != "cohesion"]
    parameter_names_with_coh = ["coh"] + parameter_names

    gof_components = input_config_dict["gof_components"].strip().split(",")
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

    gof_components = gof_weights.keys()
    # Normalize the weights so they sum to 1
    total_weight = sum(gof_weights.values())
    gof_weights = {k: v / total_weight for k, v in gof_weights.items()}
    print(gof_weights)

    gof_component_to_name = {}
    gof_component_to_name["slip_distribution"] = "gof_slip"
    gof_component_to_name["teleseismic_body_wf"] = "gof_tel"
    gof_component_to_name["teleseismic_surface_wf"] = "gof_sw"
    gof_component_to_name["regional_wf"] = "gof_reg"
    gof_component_to_name["moment_rate_function"] = "gof_MRF"
    gof_component_to_name["fault_offsets"] = "gof_offsets"
    gof_component_to_name["slip_rate"] = "gof_slip_rate"
    gof_component_to_name["seismic_moment"] = "gof_M0"

    if os.path.exists("derived_config.yaml"):
        with open("derived_config.yaml", "r") as f:
            config_dict = yaml.safe_load(f)
        refMRFfile = config_dict["reference_STF"]
    elif os.path.exists(fn):
        # for backwards compatibility
        with open("tmp/reference_STF.txt", "r") as fid:
            refMRFfile = fid.read().strip()
    else:
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

    if os.path.exists("gof_slip.pkl"):
        gofa = pickle.load(open("gof_slip.pkl", "rb"))
        gofa["sim_id"] = gofa["faultfn"].str.extract(r"dyn[/_-]([^_]+)_")[0].astype(int)
        gofa = gofa[["gof_slip", "sim_id"]]
        result_df = pd.merge(result_df, gofa, on="sim_id", how="left")

    else:
        print("gof_slip.pkl could not be found")

    if os.path.exists("rms_offset.csv"):
        gofa = pd.read_csv("rms_offset.csv", sep=",")
        gofa["sim_id"] = gofa["faultfn"].str.extract(r"dyn[/_-]([^_]+)_")[0].astype(int)
        gofa["gof_offsets"] = np.exp(-gofa["offset_rms"])
        gofa = gofa[["gof_offsets", "sim_id"]]
        result_df = pd.merge(result_df, gofa, on="sim_id", how="left")
    else:
        print("rms_offset.csv could not be found")

    if os.path.exists("area_max_R.csv"):
        gofa = pd.read_csv("area_max_R.csv", sep=",")
        gofa["sim_id"] = (
            gofa["faultfn"].str.extract(r"fault[/_-]([^_]+)_")[0].astype(int)
        )
        gofa = gofa[["area_max_R", "sim_id"]]
        result_df = pd.merge(result_df, gofa, on="sim_id", how="left")
        result_df["area_max_R"] = result_df["area_max_R"].round(1)
    else:
        print("area_max_R.csv could not be found")

    fn = "rms_slip_rate.csv"
    if os.path.exists(fn):
        gofa = pd.read_csv(fn, sep=",")
        gofa["sim_id"] = (
            gofa["fault_receiver_fname"].str.extract(r"dyn[/_-]([^_]+)_")[0].astype(int)
        )
        gofa["gof_slip_rate"] = np.exp(-gofa["slip_rate_rms"])
        gofa = gofa[["gof_slip_rate", "sim_id"]]
        result_df = pd.merge(result_df, gofa, on="sim_id", how="left")
    else:
        print(f"{fn} could not be found")

    pkl_file = "percentage_supershear.pkl"
    if os.path.exists(pkl_file):
        gofa = pickle.load(open(pkl_file, "rb"))
        gofa["sim_id"] = gofa["faultfn"].str.extract(r"dyn[/_-]([^_]+)_")[0].astype(int)
        gofa = gofa[["supershear", "sim_id"]]
        result_df = pd.merge(result_df, gofa, on="sim_id", how="left")
    else:
        print(f"{pkl_file} could not be found")

    for waveform_type in ["regional", "teleseismic"]:
        fname = f"gof_{waveform_type}_waveforms_average.pkl"
        print(fname)
        if os.path.exists(fname):
            print(f"{fname} detected: merging with results dataframe")
            gofa = pickle.load(open(fname, "rb"))
            gofa = gofa[~gofa["source_file"].str.contains(r"kinmod")]
            gofa["sim_id"] = (
                gofa["source_file"].str.extract(r"dyn[/_-]([^_]+)_")[0].astype(int)
            )
            if waveform_type == "regional":
                patterns = [
                    r"generic_ENZ",
                    r"surface_waves_ENZ\d+",
                    # example only the N component is plotted
                    r"generic_",
                    r"surface_waves_.*",
                ]

                for pattern in patterns:
                    filtered = gofa[gofa["gofa_name"].str.contains(pattern, regex=True)]
                    if not filtered.empty:
                        gofa = filtered
                        break

                gofa = gofa[["gofa", "sim_id"]]
                gofa = gofa.rename(columns={"gofa": "gof_reg"})
            else:
                gofaP = gofa[gofa["gofa_name"].str.contains(r"P_Z\d+")]
                gofaP = gofaP.rename(columns={"gofa": "gof_P"})
                # print(gofaP)
                gofaSH = gofa[gofa["gofa_name"].str.contains(r"SH_T\d+")]
                gofaSH = gofaSH.rename(columns={"gofa": "gof_SH"})
                gofaP = pd.merge(gofaP, gofaSH, on="sim_id")

                gofaSW = gofa[gofa["gofa_name"].str.contains(r"generic_ENZ\d+")]
                gofaSW = gofaSW.rename(columns={"gofa": "gof_sw"})
                if not gofaSW.empty:
                    gofaP = pd.merge(gofaP, gofaSW, on="sim_id")

                # 50% weight for SH (as in wasp)
                gofaP["gof_tel_wf"] = (gofaP["gof_P"] + 0.5 * gofaP["gof_SH"]) / 1.5
                if gofaSW.empty:
                    gofa = gofaP[["gof_P", "gof_SH", "gof_tel_wf", "sim_id"]]
                else:
                    gofa = gofaP[["gof_P", "gof_SH", "gof_tel_wf", "gof_sw","sim_id"]]
            # merge the two DataFrames by sim_id
            result_df = pd.merge(result_df, gofa, on="sim_id", how="left")

    if "gof_tel_wf" in result_df.keys():
        plot_gof_xy(result_df, "gof_tel_wf", "gof_MRF")
        plot_gof_xy(result_df, "gof_tel_wf", "duration")

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
                "{comp} given in 'gof_components' ({gof_components})"
                "but {col_name} not found in result_df"
            )
    if sum_weights != 0.0:
        result_df["combined_gof"] /= sum_weights
        component_used = {k: v * sum_weights for k, v in component_used.items()}

    result_df = result_df.sort_values(by="combined_gof", ascending=False).reset_index(
        drop=True
    )

    result_df["faultfn"] = result_df["faultfn"].apply(
        lambda x: os.path.basename(x)
        .split("_extracted-fault.xdmf")[0]
        .split("-fault.xdmf")[0]
    )
    result_df.to_pickle("compiled_results.pkl")
    print(result_df.to_string())

    required_keys = {"coh", "B", Cname, "R0"}
    if required_keys.issubset(result_df.columns):
        coh = result_df["coh"].values
        B = result_df["B"].values
        C = result_df[Cname].values
        R = result_df["R0"].values

        def are_all_elements_same(arr):
            if arr.size == 0:
                return True
            return np.all([np.array_equal(x, arr[0]) for x in arr])

        assert len(result_df) > 0
        if are_all_elements_same(coh):
            generate_BCR_plots(B, C, R)
            generate_BCR_moment_plots(B, C, R)

    varying_param = {}
    for name in parameter_names_with_coh:
        varying_param[name] = len(np.unique(result_df[name].values)) > 1

    combined_gof = result_df["combined_gof"].values
    Mw = result_df["Mw"].values
    indices_of_nlargest_values = result_df["combined_gof"].nlargest(args.nmin).index
    indices_of_nmax_largest_values = result_df["combined_gof"].nlargest(args.nmax).index
    indices_greater_than_threshold = result_df[
        result_df["combined_gof"] > args.gof_threshold
    ].index
    if len(indices_greater_than_threshold) > args.nmax:
        selected_indices = indices_of_nmax_largest_values
    else:
        selected_indices = indices_greater_than_threshold

    for fn in energy_files:
        prefix_to_match = os.path.basename(fn.split("-energy.csv")[0])
        row_with_prefix = result_df[
            result_df["faultfn"].str.startswith(prefix_to_match)
        ]
        if not row_with_prefix.empty:
            i = row_with_prefix.index[0]
        else:
            raise ValueError(
                f"could not associate {fn} ({prefix_to_match}) with a ",
                "fault filename from",
                result_df["faultfn"],
            )
            continue

        df = pd.read_csv(fn)
        df = df.pivot_table(index="time", columns="variable", values="measurement")
        dt = df.index[1] - df.index[0]
        assert dt == 0.25
        df["seismic_moment_rate"] = np.gradient(df["seismic_moment"], dt)

        if one_model_shown:
            label = "simulation"
        else:
            label = ""
            for name, is_varying in varying_param.items():
                if is_varying:
                    value = result_df[name].values[i]
                    vname = r"$\sigma_n$" if name == "sigman" else name
                    label += f"{vname}={value},"
            # remove the last ,
            label = label[0:-1]
        if i in selected_indices or i in indices_of_nlargest_values:
            if one_model_shown:
                labelargs = {"label": f"{label} (Mw={Mw[i]:.2f})"}
            else:
                labelargs = {
                    "label": f"{label} (Mw={Mw[i]:.2f}, gof={combined_gof[i]:.2})"
                }
            alpha = 1.0
        else:
            labelargs = {"color": "lightgrey", "zorder": 1}
            alpha = 0.5
        ax.plot(
            df.index.values,
            df["seismic_moment_rate"] / 1e19,
            alpha=alpha,
            **labelargs,
        )

    if refMRFfile != "tmp/moment_rate.mr":
        ax.plot(
            mr_ref[:, 0],
            mr_ref[:, 1] / 1e19,
            label=f"{ref_name} (Mw={Mwref:.2f})",
            color="black",
        )

    if ref_name != "usgs" and os.path.exists("tmp/moment_rate.mr"):
        mr_usgs = read_usgs_moment_rate("tmp/moment_rate.mr")
        mr_usgs = trim_trailing_zero(mr_usgs)
        M0usgs, Mwusgs = computeMw("usgs", mr_usgs[:, 0], mr_usgs[:, 1])
        ax.plot(
            mr_usgs[:, 0],
            mr_usgs[:, 1] / 1e19,
            label=f"usgs (Mw={Mwusgs:.2f})",
            color="k",
            linestyle="--",
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

    col = 1 if args.nmax < 8 else 2
    kargs = {"bbox_to_anchor": (1.0, 1.28)}
    ax.legend(frameon=False, loc="upper right", ncol=col, fontsize=ps, **kargs)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_ylabel(r"moment rate (e19 $\times$ Nm/s)")
    ax.set_xlabel("time (s)")

    fn = f"plots/moment_rate.{args.extension}"
    fig.savefig(fn, bbox_inches="tight", transparent=True)
    print(f"done write {fn}")
    full_path = os.path.abspath(fn)
    print(f"full path: {full_path}")
    print(f"components used for combined_gof: {component_used}")
