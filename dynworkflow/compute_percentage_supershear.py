#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import argparse
import glob
import os

import easi
import numpy as np
import pandas as pd
import pyvista as pv
import tqdm
from seissolxdmf import seissolxdmf


def generate_pv_mesh(fname):
    # Load your data
    sx = seissolxdmf(fname)

    # Create the mesh
    points = sx.ReadGeometry()
    connectivity = sx.ReadConnect().astype(np.int64)
    n_elements = sx.ReadNElements()

    faces = np.hstack(
        [np.full((n_elements, 1), 3, dtype=np.int64), connectivity]
    ).ravel()
    mesh = pv.PolyData(points, faces)

    # Read and add data
    vr = sx.ReadData("Vr", sx.ndt - 1)
    mesh.cell_data["vr"] = vr

    asl = sx.ReadData("ASl", sx.ndt - 1)
    mesh.cell_data["asl"] = asl

    mesh = mesh.threshold(value=0.05, scalars="asl")
    return mesh


def evaluate_vs(mesh, material_file):
    cell_centers = mesh.cell_centers().points
    regions = np.ones((mesh.n_cells,))
    out = easi.evaluate_model(cell_centers, regions, ["rho", "mu"], material_file)
    vs = np.sqrt(out["mu"] / out["rho"])
    return vs


def get_supershear_ratio(mesh, vs):
    mask = mesh.cell_data["vr"] > vs
    mesh.cell_data["vr_gt_vs"] = mask.astype(np.float32)
    total_area_mesh = (
        mesh.compute_cell_sizes(length=False, volume=False, area=True)
        .cell_data["Area"]
        .sum()
    )
    active_cells = mesh.threshold(0.5, scalars="vr_gt_vs")

    # Find connected components
    connected = active_cells.connectivity()

    # Now, each connected region is labeled in 'RegionId'
    region_ids = connected.cell_data["RegionId"]

    # Sum areas for each region
    areas = {}
    for region in range(region_ids.min(), region_ids.max() + 1):
        submesh = connected.extract_cells(region_ids == region)
        total_area = (
            submesh.compute_cell_sizes(length=False, volume=False, area=True)
            .cell_data["Area"]
            .sum()
        )
        areas[region] = total_area

    sum_selected_area = 0.0
    # Filter regions with more than 5 cells
    for region, area in areas.items():
        n_cells = (region_ids == region).sum()
        if n_cells > 5:
            sum_selected_area += area
    return 100 * sum_selected_area / total_area_mesh


def compute_supershear_percentile(folder, material_file):
    if os.path.exists(args.output_folder):
        args.output_folder += "/"
    fault_output_files = sorted(glob.glob(f"{folder}*-fault.xdmf"))
    results = {
        "faultfn": [],
        "supershear": [],
    }

    print("Warning: assuming region 1 for all cells when evaluating mu with easi")
    for fo in tqdm.tqdm(fault_output_files):
        if "dyn-kinmod" in fo:
            supershear_percentile = np.nan
        else:
            mesh = generate_pv_mesh(fo)
            vs = evaluate_vs(mesh, material_file)
            supershear_percentile = get_supershear_ratio(mesh, vs)
        results["faultfn"].append(fo)
        results["supershear"].append(supershear_percentile)
    df = pd.DataFrame(results)
    print(df)
    fname = "percentage_supershear.pkl"
    df.to_pickle(fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""compute percentage of supershear in slip area for an
        ensemble of DR models. all on the same mesh.
        partitionning may differ though"""
    )
    parser.add_argument("output_folder", help="folder where the models lie")
    parser.add_argument("material_file", help="easi yaml file defining rho mu lambda")

    args = parser.parse_args()
    compute_supershear_percentile(args.output_folder, args.material_file)
