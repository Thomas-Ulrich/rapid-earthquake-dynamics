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


def plot_combined_gof_plot(df, keys_to_plot, nlines, preferred_model):
    print(df.keys())

    label_map = {
        "gof_offsets": "Fault-offsets (GOF)",
        "gof_slip_rate": "Slip-rate at CCTV (GOF)",
        "gof_slip": "Fault-slip distribution (GOF)",
        "gof_body_wf": "Body waveforms (GOF)",
        "gof_reg": "Regional waveforms  (GOF)",
        "gof_MRF": "Moment-rate function (GOF)",
        "gof_surf_wf": "Surface waveforms (GOF)",
        "combined_gof": "Combined GOF",
    }

    # Check if all keys exist in the DataFrame
    missing_keys = set(keys_to_plot) - set(df.columns)
    assert not missing_keys, f"Missing required keys in DataFrame: {missing_keys}"
    ncol = int(np.ceil(len(keys_to_plot) / nlines))

    if "sigman" in df.columns:
        dim_var_x = {"col": "sigman", "label": r"$\sigma_n$"}
    elif "R" in df.columns:
        dim_var_x = {"col": "R", "label": "R"}
    else:
        raise ValueError("structure of df not understood")

    for B in df["B"].unique():
        fig, ax = plt.subplots(nlines, ncol, figsize=(4 * ncol, 3 * nlines), dpi=80)

        dim_vars_0 = {
            "x": dim_var_x,
            "y": {"col": "C", "label": "C"},
            "z": {"col": "B", "label": "B"},
        }

        for i in range(nlines):
            for j in range(ncol):
                k = i * ncol + j
                if k > len(keys_to_plot):
                    ax[i, j].set_visible(False)
                    break
                dim_vars = copy.deepcopy(dim_vars_0)
                dim_vars["x"]["label"] = (
                    None if i < nlines - 1 else dim_vars_0["x"]["label"]
                )
                dim_vars["y"]["label"] = None if j > 0 else dim_vars_0["y"]["label"]
                contour_lines = None
                key = keys_to_plot[k]
                label = label_map[key] if key in label_map.keys() else key
                dim_vars["v"] = {"col": key, "label": label}
                alpha = "abcdefghij"
                letter = alpha[k]
                ax[i, j].set_title(f"{letter}.", fontweight="bold")
                plot_xy_panel(
                    fig,
                    ax[i, j],
                    df,
                    dim_vars,
                    val_z=B,
                    cmap=cm.cmaps["lipari_r"],
                    contour_lines=contour_lines,
                )
                if B == preferred_model["B"]:
                    cx, cy = dim_vars["x"]["col"], dim_vars["y"]["col"]
                    ax[i, j].scatter(
                        [preferred_model[cx]], [preferred_model[cy]], c="g", marker="x"
                    )
                    print(
                        "plotting preferred",
                        [preferred_model[cx]],
                        [preferred_model[cy]],
                        B,
                    )

        ax[0, 0].set_title(f"a. B={B}", fontweight="bold")
        ext = "pdf"
        fn = f"plots/figure_panelsB{B}_gof.{ext}"
        plt.savefig(fn)
        print(f"done writing {fn}")
