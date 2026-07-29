import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
import pandas as pd
import copy
import os
import matplotlib

from dynworkflow.stf_loader import read_usgs_moment_rate, trim_trailing_zero


ps = 12
matplotlib.rcParams.update({"font.size": ps})
plt.rcParams["font.family"] = "sans"
matplotlib.rc("xtick", labelsize=ps)
matplotlib.rc("ytick", labelsize=ps)


def plot_xy_panel(
    fig,
    ax,
    df: pd.DataFrame,
    dim_vars: dict,
    val_z,
    cmap,
    plot_type: str = "contourf",
    contour_lines: list | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Generate a 2D spatial/parameter plot (X vs Y) filtered at a slice Z = val_z.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Reference to parent figure (for colorbar placement).
    ax : matplotlib.axes.Axes
        Target subplot axis.
    df : pd.DataFrame
        DataFrame containing simulation parameters and GoF metrics.
    dim_vars : dict
        Mapping dictionary structured as:
        {
            "x": {"col": "B", "label": "B-value"},
            "y": {"col": "C", "label": "C-value"},
            "z": {"col": "R0", "label": "R0"},
            "v": {"col": "combined_gof", "label": "Combined GoF"}
        }
    val_z : float/int
        Value to filter the 'z' column by.
    cmap : Colormap
        Matplotlib colormap.
    plot_type : str, optional
        'contourf' for filled contours or 'pcolormesh' for discrete cells.
    contour_lines : list of float, optional
        Levels at which to draw labeled contour lines.
    vmin : float, optional
        Minimum value for the colorbar scale.
    vmax : float, optional
        Maximum value for the colorbar scale.
    """
    col_x = dim_vars["x"]["col"]
    col_y = dim_vars["y"]["col"]
    col_z = dim_vars["z"]["col"]
    col_v = dim_vars["v"]["col"]

    # 1. Filter DataFrame by z-slice
    sub_df = df[df[col_z] == val_z]

    if sub_df.empty:
        ax.set_title(f"{col_z}={val_z} (No Data)")
        return

    # 2. Vectorized 2D grid generation via pivot table
    pivot = sub_df.pivot_table(index=col_y, columns=col_x, values=col_v)
    X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
    values = pivot.values.astype(float)

    # 3. Determine color limits (vmin/vmax)
    if vmin is not None and vmax is not None and vmin == vmax:
        vmax = vmin + 1e-6

    # 4. Plot surface (contourf or pcolormesh)
    if plot_type == "contourf":
        if min(values.shape) == 1:
            return
        if vmin is not None and vmax is not None:
            levels = np.linspace(vmin, vmax, 21)
            im = ax.contourf(
                X, Y, values, cmap=cmap, levels=levels, vmin=vmin, vmax=vmax
            )
        else:
            im = ax.contourf(X, Y, values, cmap=cmap, levels=20)
        if contour_lines:
            contours = ax.contour(
                X, Y, values, levels=contour_lines, colors="k", linestyles="-"
            )
            ax.clabel(contours, inline=True, fontsize=9, fmt="%g")
    else:
        if vmin is not None and vmax is not None:
            im = ax.pcolormesh(
                X, Y, values, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax
            )
        else:
            im = ax.contourf(X, Y, values, cmap=cmap, levels=20)

    # Limit bounds to exact pivot extents
    ax.set_xlim(pivot.columns.min(), pivot.columns.max())
    ax.set_ylim(pivot.index.min(), pivot.index.max())

    # 5. Hatch invalid / NaN cells
    mask_invalid = np.isnan(values)
    if np.any(mask_invalid):
        ax.pcolor(
            X, Y, np.ma.masked_where(~mask_invalid, mask_invalid), hatch="//", alpha=0
        )

    # 6. Ticks and Labels Formatting
    ax.set_xticks(pivot.columns.values)
    ax.set_yticks(pivot.index.values)

    # X-Label handling
    x_label = dim_vars["x"].get("label", col_x)
    if x_label and dim_vars["x"].get("show_label", True):
        ax.set_xlabel(x_label)
        vals = pivot.columns.values
        if len(vals) > 7:
            ax.set_xticklabels(
                [f"{x:g}" if i % 2 == 0 else "" for i, x in enumerate(vals)]
            )
        else:
            ax.set_xticklabels([f"{x:g}" for x in vals])
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])

    # Y-Label handling
    y_label = dim_vars["y"].get("label", col_y)
    if y_label:
        ax.set_ylabel(y_label)
        ax.set_yticklabels([f"{y:g}" for y in pivot.index.values])
    else:
        ax.set_yticklabels([])

    title_z = dim_vars["z"].get("label", col_z)
    ax.set_title(f"{title_z}={val_z}")

    # 7. Colorbar attachment
    v_label = dim_vars["v"].get("label", col_v)
    if "GOF" in v_label:
        fig.colorbar(im, ax=ax, label=v_label, format="%.2f")
    else:
        fig.colorbar(im, ax=ax, label=v_label)


def plot_combined_gof_plot(
    df: pd.DataFrame,
    keys_to_plot: list[str],
    nlines: int,
    preferred_model: dict,
    combine_B_in_one_fig: bool = False,
    share_colorbar: bool = True,
    extra_label_map: dict | None = None,
    output_prefix: str | None = None,
):
    """
    Plot Goodness-of-Fit (GoF) panels across parameter slices.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe containing metrics and simulation parameters.
    keys_to_plot : list[str]
        DataFrame columns to visualize per B-value.
    nlines : int
        If combine_B_in_one_fig=False: Number of rows PER figure (per B-value).
        If combine_B_in_one_fig=True: Total number of rows in the single combined
        figure.
    preferred_model : dict
        Dictionary with preferred parameters, e.g. {'B': 0.9, 'C': 0.1, 'R': 0.7}.
    combine_B_in_one_fig : bool, optional
        If True, combines all B-values into one figure respecting `nlines` as total
        rows.
        If False, exports one individual figure per B-value.
    share_colorbar : bool, optional
        If True, precomputes global (vmin, vmax) bounds across the whole DataFrame for
        each metric in `keys_to_plot` so color scales match across B-values. Default
        is True.
    extra_label_map : dict, optional
        Custom mapping dictionary to add or override default panel labels.
    output_prefix : str, optional
            Custom filename or path prefix for exported PDF(s).
            - If combine_B_in_one_fig=True:
              Defaults to "plots/figure_panels_allB_gof.pdf".
              If provided as "my_run", saves to "plots/my_run.pdf".
            - If combine_B_in_one_fig=False:
              Defaults to "plots/figure_panelsB{B}_gof.pdf".
              If provided as "my_run", saves to "plots/my_run_B{B}.pdf".
    """
    # 1. Validation
    missing_keys = set(keys_to_plot) - set(df.columns)
    assert not missing_keys, f"Missing required keys in DataFrame: {missing_keys}"

    default_label_map = {
        "gof_offsets": "Fault-offsets (GOF)",
        "gof_slip_rate": "Slip-rate at CCTV (GOF)",
        "gof_slip": "Fault-slip distribution (GOF)",
        "gof_body_wf": "Body waveforms (GOF)",
        "gof_reg": "Regional waveforms (GOF)",
        "gof_MRF": "Moment-rate function (GOF)",
        "gof_surf_wf": "Surface waveforms (GOF)",
        "combined_gof": "Combined GOF",
    }
    label_map = default_label_map | (extra_label_map or {})

    unique_B = sorted(df["B"].unique())
    n_B = len(unique_B)
    num_keys = len(keys_to_plot)

    # 2. Precompute Global Bounds
    if share_colorbar:
        gof_bounds = {
            key: (df[key].min(), df[key].max())
            for key in keys_to_plot
            if key in df.columns
        }
    else:
        gof_bounds = {}

    if "sigman" in df.columns:
        dim_var_x = {"col": "sigman", "label": r"$\sigma_\mathrm{n}$"}
    elif "R" in df.columns:
        dim_var_x = {"col": "R", "label": "R"}
    elif "Ru" in df.columns:  # for Myanmar study
        dim_var_x = {"col": "Ru", "label": r"$R_\mathrm{u}$"}
    else:
        raise ValueError(
            "Structure of df not understood (neither 'sigman' nor 'R' found)"
        )

    dim_vars_0 = {
        "x": dim_var_x,
        "y": {"col": "C", "label": "C"},
        "z": {"col": "B", "label": "B"},
    }
    alpha = "abcdefghijklmnopqrstuvwxyz"

    # 3. Combined Canvas Mode
    if combine_B_in_one_fig:
        total_subpanels = n_B * num_keys
        ncol = int(np.ceil(total_subpanels / nlines))

        fig, ax_all = plt.subplots(
            nlines, ncol, figsize=(4 * ncol, 3 * nlines), dpi=80, squeeze=False
        )
        axes_flat = ax_all.flat
        panel_counter = 0

        for b_idx, B in enumerate(unique_B):
            for k, key in enumerate(keys_to_plot):
                target_ax = axes_flat[panel_counter]
                col_j = panel_counter % ncol

                dim_vars = copy.deepcopy(dim_vars_0)
                dim_vars["x"]["show_label"] = panel_counter + ncol >= total_subpanels
                dim_vars["y"]["label"] = None if col_j > 0 else dim_vars_0["y"]["label"]

                label = label_map.get(key, key)
                dim_vars["v"] = {"col": key, "label": label}

                letter = (
                    alpha[panel_counter]
                    if panel_counter < len(alpha)
                    else f"{panel_counter + 1}"
                )
                vmin, vmax = gof_bounds.get(key, (None, None))
                plot_xy_panel(
                    fig,
                    target_ax,
                    df,
                    dim_vars,
                    val_z=B,
                    cmap=cm.cmaps["lipari_r"],
                    contour_lines=None,
                    vmin=vmin,
                    vmax=vmax,
                )

                title_str = f"{letter}. B={B}" if k == 0 else f"{letter}."
                target_ax.set_title(title_str, fontweight="bold")

                if preferred_model and B == preferred_model.get("B"):
                    cx, cy = dim_vars["x"]["col"], dim_vars["y"]["col"]
                    target_ax.scatter(
                        [preferred_model[cx]],
                        [preferred_model[cy]],
                        c="g",
                        marker="x",
                    )

                panel_counter += 1

        for unused_idx in range(panel_counter, len(axes_flat)):
            axes_flat[unused_idx].set_visible(False)

        plt.tight_layout()
        if output_prefix:
            # Ensures .pdf extension is appended cleanly
            fn = (
                output_prefix
                if output_prefix.endswith(".pdf")
                else f"plots/{output_prefix}.pdf"
            )
        else:
            fn = "plots/figure_panels_allB_gof.pdf"
        fig.savefig(fn, bbox_inches="tight")
        plt.close(fig)
        print(f"done writing {fn}")

    # 4. Individual Figure per B-value Mode
    else:
        ncol = int(np.ceil(num_keys / nlines))

        for B in unique_B:
            fig, ax_grid = plt.subplots(
                nlines, ncol, figsize=(4 * ncol, 3 * nlines), dpi=80, squeeze=False
            )

            for i in range(nlines):
                for j in range(ncol):
                    k = i * ncol + j
                    target_ax = ax_grid[i, j]

                    if k >= num_keys:
                        target_ax.set_visible(False)
                        continue

                    dim_vars = copy.deepcopy(dim_vars_0)
                    dim_vars["x"]["show_label"] = k + ncol >= num_keys
                    dim_vars["y"]["label"] = None if j > 0 else dim_vars_0["y"]["label"]

                    key = keys_to_plot[k]
                    label = label_map.get(key, key)
                    dim_vars["v"] = {"col": key, "label": label}

                    letter = alpha[k] if k < len(alpha) else f"{k + 1}"
                    vmin, vmax = gof_bounds.get(key, (None, None))

                    plot_xy_panel(
                        fig,
                        target_ax,
                        df,
                        dim_vars,
                        val_z=B,
                        cmap=cm.cmaps["lipari_r"],
                        contour_lines=None,
                        vmin=vmin,
                        vmax=vmax,
                    )

                    target_ax.set_title(f"{letter}.", fontweight="bold")

                    if preferred_model and B == preferred_model.get("B"):
                        cx, cy = dim_vars["x"]["col"], dim_vars["y"]["col"]
                        target_ax.scatter(
                            [preferred_model[cx]],
                            [preferred_model[cy]],
                            c="g",
                            marker="x",
                        )

            ax_grid[0, 0].set_title(f"a. B={B}", fontweight="bold")
            plt.tight_layout()
            if output_prefix:
                # Strips .pdf if included, then appends the B-value
                base_prefix = (
                    output_prefix[:-4]
                    if output_prefix.endswith(".pdf")
                    else output_prefix
                )
                fn = f"plots/{base_prefix}_B{B}.pdf"
            else:
                fn = f"plots/figure_panelsB{B}_gof.pdf"

            fn = f"plots/figure_panelsB{B}_gof.pdf"
            fig.savefig(fn, bbox_inches="tight")
            plt.close(fig)
            print(f"done writing {fn}")


def plot_moment_rates(
    energy_files: list[str],
    result_df: pd.DataFrame,
    args,
    ref_stfs: list,
    mr_ref: np.ndarray,
    ref_name: str,
    Mwref: float,
    varying_param: dict,
):
    """Generates and saves the moment rate plot."""
    one_model_shown = args.nmax == 1
    matplotlib.rcParams["lines.linewidth"] = 0.5 if one_model_shown else 1.0

    centimeter = 1 / 2.54
    figsize = (7.5 * centimeter, 4.0 * centimeter) if one_model_shown else (8, 4)
    fig = plt.figure(figsize=figsize, dpi=80)
    ax = fig.add_subplot(111)

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
            label_parts = []
            for name, is_varying in varying_param.items():
                if is_varying:
                    value = result_df[name].values[i]
                    vname = r"$\sigma_n$" if name == "sigman" else name
                    unit = " MPa" if name == "sigman" else ""
                    label_parts.append(f"{vname}={value}{unit}")
            label = ", ".join(label_parts)

        if i in selected_indices or i in indices_of_nlargest_values:
            if one_model_shown:
                labelargs = {"label": f"{label} (Mw={Mw[i]:.2f})"}
            else:
                labelargs = {
                    "label": f"{label} (Mw={Mw[i]:.2f}, GOF={combined_gof[i]:.2})"
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

    refMRFfile = ref_stfs[0][0] if ref_stfs else ""
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
        M0usgs = np.trapezoid(mr_usgs[:, 1], x=mr_usgs[:, 0])
        Mwusgs = 2.0 * np.log10(M0usgs) / 3.0 - 6.07
        ax.plot(
            mr_usgs[:, 0],
            mr_usgs[:, 1] / 1e19,
            label=f"USGS (Mw={Mwusgs:.2f})",
            color="k",
            linestyle="--",
        )

    ls = ["-", ":", "-."]
    if len(ref_stfs) > 1:
        for kkk, mrfdata in enumerate(ref_stfs[1:]):
            mrf_file, mrf_label = mrfdata
            if not os.path.exists(mrf_file):
                print(f"Warning: file {mrf_file} does not exist, skipping.")
                continue

            ext = os.path.splitext(mrf_file.lower())[1]
            if ext == ".csv":
                mrf = np.genfromtxt(mrf_file, delimiter=",", skip_header=1)
            else:
                mrf = np.loadtxt(mrf_file)

            M0 = np.trapezoid(mrf[:, 1], x=mrf[:, 0])
            Mw_mrf = 2.0 * np.log10(M0) / 3.0 - 6.07
            ax.plot(
                mrf[:, 0],
                mrf[:, 1] / 1e19,
                label=f"{mrf_label} (Mw={Mw_mrf:.2f})",
                color="k",
                linestyle=ls[kkk % len(ls)],
            )

    col = 1 if args.nmax < 8 else 2
    kargs = {"bbox_to_anchor": (1.0, 1.28)}
    ax.legend(
        frameon=False, loc="upper right", ncol=col, fontsize=args.font_size[0], **kargs
    )
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # ax.set_ylabel(r"Moment rate (e19 $\times$ Nm/s)")
    ax.set_ylabel(r"Moment rate ($\times 10^{19}$ Nm/s)")
    ax.set_xlabel("Time (s)")

    fn_out = f"plots/moment_rate.{args.extension}"
    fig.savefig(fn_out, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"done write {fn_out}")
    print(f"full path: {os.path.abspath(fn_out)}")
