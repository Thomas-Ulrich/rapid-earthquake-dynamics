import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
import pandas as pd
import copy


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
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)

    if vmin == vmax:
        vmax = vmin + 1e-6

    # 4. Plot surface (contourf or pcolormesh)
    if plot_type == "contourf":
        levels = np.linspace(vmin, vmax, 21)
        im = ax.contourf(X, Y, values, cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
        if contour_lines:
            contours = ax.contour(
                X, Y, values, levels=contour_lines, colors="k", linestyles="-"
            )
            ax.clabel(contours, inline=True, fontsize=9, fmt="%1.2f")
    else:
        im = ax.pcolormesh(
            X, Y, values, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax
        )

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
    if x_label:
        ax.set_xlabel(x_label)
        vals = pivot.columns.values
        if len(vals) > 7:
            ax.set_xticklabels(
                [f"{x:g}" if i % 2 == 0 else "" for i, x in enumerate(vals)]
            )
        else:
            ax.set_xticklabels([f"{x:g}" for x in vals])
    else:
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
    fig.colorbar(im, ax=ax, label=v_label)


def plot_combined_gof_plot(
    df: pd.DataFrame,
    keys_to_plot: list[str],
    nlines: int,
    preferred_model: dict,
    combine_B_in_one_fig: bool = False,
    share_colorbar: bool = True,
    extra_label_map: dict | None = None,
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
        dim_var_x = {"col": "sigman", "label": r"$\sigma_n$"}
    elif "R" in df.columns:
        dim_var_x = {"col": "R", "label": "R"}
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
                row_i = panel_counter // ncol
                col_j = panel_counter % ncol

                dim_vars = copy.deepcopy(dim_vars_0)
                dim_vars["x"]["label"] = (
                    None if row_i < nlines - 1 else dim_vars_0["x"]["label"]
                )
                dim_vars["y"]["label"] = None if col_j > 0 else dim_vars_0["y"]["label"]

                label = label_map.get(key, key)
                dim_vars["v"] = {"col": key, "label": label}

                letter = (
                    alpha[panel_counter]
                    if panel_counter < len(alpha)
                    else f"{panel_counter+1}"
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
                    dim_vars["x"]["label"] = (
                        None if i < nlines - 1 else dim_vars_0["x"]["label"]
                    )
                    dim_vars["y"]["label"] = None if j > 0 else dim_vars_0["y"]["label"]

                    key = keys_to_plot[k]
                    label = label_map.get(key, key)
                    dim_vars["v"] = {"col": key, "label": label}

                    letter = alpha[k] if k < len(alpha) else f"{k+1}"
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
            fn = f"plots/figure_panelsB{B}_gof.pdf"
            fig.savefig(fn, bbox_inches="tight")
            plt.close(fig)
            print(f"done writing {fn}")
