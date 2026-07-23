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

    # 3. Plot surface (contourf or pcolormesh)
    if plot_type == "contourf":
        im = ax.contourf(X, Y, values, cmap=cmap, levels=20)
        if contour_lines:
            contours = ax.contour(
                X, Y, values, levels=contour_lines, colors="k", linestyles="-"
            )
            ax.clabel(contours, inline=True, fontsize=9, fmt="%1.2f")
    else:
        im = ax.pcolormesh(X, Y, values, cmap=cmap, shading="auto")

    # Limit bounds to exact pivot extents
    ax.set_xlim(pivot.columns.min(), pivot.columns.max())
    ax.set_ylim(pivot.index.min(), pivot.index.max())

    # 4. Hatch invalid / NaN cells
    mask_invalid = np.isnan(values)
    if np.any(mask_invalid):
        ax.pcolor(
            X, Y, np.ma.masked_where(~mask_invalid, mask_invalid), hatch="//", alpha=0
        )

    # 5. Ticks and Labels Formatting
    ax.set_xticks(pivot.columns.values)
    ax.set_yticks(pivot.index.values)

    # X-Label handling (with decimation for dense grids)
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

    # 6. Colorbar attachment
    v_label = dim_vars["v"].get("label", col_v)
    fig.colorbar(im, ax=ax, label=v_label)


def plot_combined_gof_plot(
    df: pd.DataFrame,
    keys_to_plot: list[str],
    nlines: int,
    preferred_model: dict,
    combine_B_in_one_fig: bool = False,
):
    """
    Plot GoF panels across parameter slices.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe containing metrics and simulation parameters.
    keys_to_plot : list[str]
        DataFrame columns to visualize.
    nlines : int
        Number of rows PER B-value panel block.
    preferred_model : dict
        Dictionary with preferred parameters, e.g. {'B': 0.9, 'C': 0.1, 'R': 0.7}.
    combine_B_in_one_fig : bool, optional
        If True, plots all B-values into a single figure saved as one PDF.
        If False, exports one PDF figure per B-value.
    """
    # 1. Validation
    missing_keys = set(keys_to_plot) - set(df.columns)
    assert not missing_keys, f"Missing required keys in DataFrame: {missing_keys}"

    label_map = {
        "gof_offsets": "Fault-offsets (GOF)",
        "gof_slip_rate": "Slip-rate at CCTV (GOF)",
        "gof_slip": "Fault-slip distribution (GOF)",
        "gof_body_wf": "Body waveforms (GOF)",
        "gof_reg": "Regional waveforms (GOF)",
        "gof_MRF": "Moment-rate function (GOF)",
        "gof_surf_wf": "Surface waveforms (GOF)",
        "combined_gof": "Combined GOF",
    }

    ncol = int(np.ceil(len(keys_to_plot) / nlines))
    unique_B = sorted(df["B"].unique())
    n_B = len(unique_B)

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

    # 2. Setup Figure Canvas
    if combine_B_in_one_fig:
        total_rows = nlines * n_B
        fig, ax_global = plt.subplots(
            total_rows, ncol, figsize=(4 * ncol, 3 * total_rows), dpi=80, squeeze=False
        )
    else:
        fig = None

    # 3. Main Plotting Loop
    for b_idx, B in enumerate(unique_B):
        if not combine_B_in_one_fig:
            fig, ax_grid = plt.subplots(
                nlines, ncol, figsize=(4 * ncol, 3 * nlines), dpi=80, squeeze=False
            )
        else:
            row_offset = b_idx * nlines
            # Slice row block for current B-value
            ax_grid = ax_global[row_offset : row_offset + nlines, :]

        for i in range(nlines):
            for j in range(ncol):
                k = i * ncol + j
                target_ax = ax_grid[i, j]

                if k >= len(keys_to_plot):
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

                letter_idx = (
                    (b_idx * len(keys_to_plot) + k) if combine_B_in_one_fig else k
                )
                letter = (
                    alpha[letter_idx]
                    if letter_idx < len(alpha)
                    else f"{letter_idx + 1}"
                )

                plot_xy_panel(
                    fig,
                    target_ax,
                    df,
                    dim_vars,
                    val_z=B,
                    cmap=cm.cmaps["lipari_r"],
                    contour_lines=None,
                )

                title_prefix = (
                    f"{letter}. B={B}"
                    if (combine_B_in_one_fig and k == 0) or not combine_B_in_one_fig
                    else f"{letter}."
                )
                target_ax.set_title(title_prefix, fontweight="bold")

                if preferred_model and B == preferred_model.get("B"):
                    cx, cy = dim_vars["x"]["col"], dim_vars["y"]["col"]
                    target_ax.scatter(
                        [preferred_model[cx]],
                        [preferred_model[cy]],
                        c="g",
                        marker="x",
                    )

        # Save individual figures if not combined
        if not combine_B_in_one_fig:
            plt.tight_layout()
            fn = f"plots/figure_panelsB{B}_gof.pdf"
            fig.savefig(fn, bbox_inches="tight")
            plt.close(fig)
            print(f"done writing {fn}")

    # Save combined figure if enabled
    if combine_B_in_one_fig:
        plt.tight_layout()
        fn = "plots/figure_panels_allB_gof.pdf"
        fig.savefig(fn, bbox_inches="tight")
        plt.close(fig)
        print(f"done writing {fn}")
