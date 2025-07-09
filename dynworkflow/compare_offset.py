#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich, Mathilde Marchandon

import pandas as pd
from pyproj import Transformer
import numpy as np
import matplotlib.pylab as plt
import trimesh
import seissolxdmf
from scipy import spatial
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import re
import argparse
import glob

# plt.rc("font", family="Poppins", size=8)
plt.rc("font", size=8)


class seissolxdmfExtended(seissolxdmf.seissolxdmf):
    def compute_centers(self):
        xyz = self.ReadGeometry()
        connect = self.ReadConnect()
        centers = np.sum(xyz[connect], axis=1) / 3.0
        return centers

    def compute_strike(self):
        xyz = self.ReadGeometry()
        connect = self.ReadConnect()
        strike = np.zeros((connect.shape[0], 2))

        for i in range(connect.shape[0]):
            p0 = xyz[connect[i, 0]]
            p1 = xyz[connect[i, 1]]
            p2 = xyz[connect[i, 2]]

            # Vectors in the triangle plane
            v1 = p1 - p0
            v2 = p2 - p0

            # Normal vector to the triangle (fault) plane
            normal = np.cross(v1, v2)

            # Horizontal projection of the normal vector (zero z-component)
            horizontal_normal = np.array([normal[0], normal[1], 0.0])
            if np.linalg.norm(horizontal_normal) < 1e-8:
                raise ValueError("fault is flat (normal has no horizontal component")

            # Strike is perpendicular to horizontal normal, lying in XY plane
            strike_vec = np.cross([0, 0, 1], horizontal_normal)
            strike_vec = strike_vec[:2] / np.linalg.norm(strike_vec[:2])

            strike[i, :] = strike_vec

        return strike

    def get_fault_trace(self, threshold_z):
        geom = self.ReadGeometry()
        connect = self.ReadConnect()

        mesh = trimesh.Trimesh(vertices=geom, faces=connect)

        # Boundary edges: edges that appear exactly once (i.e., on the boundary)
        boundary_edges = mesh.edges[
            trimesh.grouping.group_rows(mesh.edges_sorted, require_count=2)
        ][:, :, 1]
        ids_external_nodes = np.unique(boundary_edges)

        # Extract boundary nodes at surface (z=0), sorted by y
        nodes = mesh.vertices[ids_external_nodes]
        nodes = nodes[nodes[:, 2] >= threshold_z]
        nodes = nodes[np.argsort(nodes[:, 1])]

        # Filter out steep (near-vertical) segments using z-gradient
        grad = np.gradient(nodes, axis=0)
        grad /= np.linalg.norm(grad, axis=1)[:, None]
        nodes = nodes[np.abs(grad[:, 2]) < 0.8]

        return nodes


def extract_dyn_number(filename):
    match = re.search(r"dyn_(\d+)", filename)
    return int(match.group(1)) if match else -1


def plot_individual_offset_figure(df, acc_dist, slip_at_trace, fname):
    plt.rc("font", size=12)
    fig = plt.figure(figsize=(7.5, 3.0))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Distance along strike (km)")
    ax.set_ylabel("Fault offsets (m)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    lw = 0.8
    ax.errorbar(
        acc_dist,
        df["offset"],
        yerr=df["error"],
        color="k",
        linestyle="-",
        linewidth=lw / 2.0,
        label="Inferred offset",
        marker="o",
        markersize=2,
    )

    ax.plot(
        acc_dist,
        slip_at_trace,
        "royalblue",
        linewidth=1,
        label="Predicted offset",
    )

    ax.text(-10, 6.5, "South", fontweight="medium")
    ax.text(500, 6.5, "North", fontweight="medium")

    plt.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"done writing {fname}")
    plt.close(fig)
    plt.rc("font", size=8)


def init_all_offsets_figure(acc_dist, df):
    "init plot with every model"

    fig = plt.figure(figsize=(7.5, 3.0))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Distance along strike (km)")
    ax.set_ylabel("Fault offsets (m)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    lw = 0.8
    ax.errorbar(
        acc_dist,
        df["offset"],
        yerr=df["error"],
        color="k",
        linestyle="-",
        linewidth=lw / 2.0,
        label="Inferred offset",
        marker="o",
        markersize=2,
        zorder=2,
    )
    ax.text(-10, 6.5, "South", fontweight="medium")
    ax.text(500, 6.5, "North", fontweight="medium")
    return fig, ax


def remove_spikes(data, threshold=5.0):
    """
    Remove spikes from a 1D numpy array using gradient thresholding.

    Parameters:
        data (np.ndarray): Input 1D array.
        threshold (float): Threshold on gradient to detect spikes.

    Returns:
        np.ndarray: Array with spikes replaced by interpolated values.
    """
    data = data.copy()
    grad = np.gradient(data)

    # Identify spike locations where gradient is abnormally large
    spike_indices = np.where(np.abs(grad) > threshold)[0]
    for idx in spike_indices:
        if idx + 2 in spike_indices:
            if 0 <= idx < len(data) - 2:
                # Replace spike with average of neighbors
                data[idx + 1] = 0.5 * (data[idx] + data[idx + 2])

    return data


def compute_rms_offset(folder, offset_data, threshold_z, individual_figures):
    # Read optical offset
    df = pd.read_csv(offset_data, sep=",")
    df = df.sort_values(by=["lat", "lon"])

    # Transform lat lon in DR models projection and compute distance
    myproj = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=95.93 +lat_0=22.00"
    transformer = Transformer.from_crs("epsg:4326", myproj, always_xy=True)
    x, y = transformer.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    xy = np.vstack((x, y)).T
    dist = np.linalg.norm(xy[1:, :] - xy[0:-1, :], axis=1)
    acc_dist = np.add.accumulate(dist) / 1e3
    acc_dist = np.insert(acc_dist, 0, 0)

    # Create figure folder
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # List all models in DR output folder
    models = sorted(glob.glob(f"{folder}*-fault.xdmf"))
    wrms = np.zeros(np.size(models))

    #################################################################
    # Compute weighted RMS and plot individual figure of each model #
    #################################################################

    slip_at_traces = []
    results = {
        "faultfn": [],
        "offset_rms": [],
    }

    fig, ax = init_all_offsets_figure(acc_dist, df)
    # Loop on each model in the output filder
    for i, fault in enumerate(models):
        base_name = os.path.basename(fault).replace("_extracted-fault.xdmf", "")

        # Read SeisSol output
        sx = seissolxdmfExtended(f"{fault}")
        fault_centers = sx.compute_centers()
        idt = sx.ReadNdt() - 1
        Sls = np.abs(sx.ReadData("Sls", idt))

        # Find indices of surface subfaults
        trace_nodes = sx.get_fault_trace(threshold_z)[::1]
        tree = spatial.KDTree(fault_centers)
        dist, idsf = tree.query(trace_nodes)
        # Surface fault slip
        slip_at_trace = Sls[idsf]

        # Surface fault slip at observation location
        tree = spatial.KDTree(trace_nodes[:, 0:2])
        dist, idsf2 = tree.query(xy)
        slip_at_trace = slip_at_trace[idsf2]

        slip_at_trace = remove_spikes(slip_at_trace, threshold=1.0)

        if individual_figures:
            if len(models) == 1:
                fname = f"figures/comparison_selected_offset.svg"
            else:
                fname = f"figures/comparison_offset_sentinel2_{base_name}.svg"
            plot_individual_offset_figure(df, acc_dist, slip_at_trace, fname)

        ax.plot(
            acc_dist,
            slip_at_trace,
            "gainsboro",
            linewidth=0.8,
            label="Predicted offset",
        )

        residuals = df["offset"] - slip_at_trace
        weights = 1 / df["error"] ** 2
        wrms[i] = np.sqrt(np.sum(weights * residuals**2) / np.sum(weights))
        slip_at_traces.append(slip_at_trace)
        print(f"Model: {base_name} RMS = {wrms[i]:.5f} m")
        results["faultfn"].append(fault)
        results["offset_rms"].append(wrms[i])

    top10_indices = np.argsort(wrms)[:10]
    cmap = cm.viridis_r
    norm = mcolors.Normalize(
        vmin=min(wrms[top10_indices]), vmax=max(wrms[top10_indices])
    )
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)

    print("10 best models:")
    for modeli in top10_indices[::-1]:
        col = cmap(norm(wrms[modeli]))
        ax.plot(
            acc_dist,
            slip_at_traces[modeli],
            color=col,
            linewidth=0.8,
            label="Predicted offset",
        )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="WRMS")
    fn = "figures/comparison_offset_all_models_10bestOffset.pdf"
    plt.savefig(fn, dpi=200, bbox_inches="tight")
    print(f"done writing {fn}")

    dfr = pd.DataFrame(results)
    dfr.to_csv("rms_offset.csv", index=True, index_label="id")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""compute fit (RMS) to offset of models from an
        ensemble of DR models."""
    )
    parser.add_argument("output_folder", help="folder where the models lie")
    parser.add_argument("offset_data", help="path to offset data")
    parser.add_argument(
        "--threshold_z",
        help="threshold depth used for selecting fault trace nodes",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--individual_figures",
        action="store_true",
        help="plot one figure for each file",
    )

    args = parser.parse_args()
    compute_rms_offset(
        args.output_folder, args.offset_data, args.threshold_z, args.individual_figures
    )
