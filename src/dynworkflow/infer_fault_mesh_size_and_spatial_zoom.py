#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import argparse
import os
import os.path

import numpy as np
from kinematic_models.multi_fault_plane import MultiFaultPlane
from sklearn.decomposition import PCA


def infer_quantities(filename, proj, mesh_size="auto"):
    prefix, ext = os.path.splitext(filename)
    prefix = os.path.basename(prefix)
    mfp = MultiFaultPlane.from_file(filename)

    dx = float("inf")
    dy = float("inf")
    min_fault_plane_area = float("inf")
    total_area = 0
    for p, p1 in enumerate(mfp.fault_planes):
        p1.compute_xy_from_latlon(proj)
        xyz = np.column_stack(
            (p1.x.flatten(), p1.y.flatten(), -p1.depth.flatten() * 1e3)
        )
        # Perform PCA to get principal axes
        pca = PCA(n_components=2)
        points = pca.fit_transform(xyz) / 1e3
        la, lb = np.amax(points, axis=0) - np.amin(points, axis=0)
        min_fault_plane_area = min(min_fault_plane_area, la * lb)
        total_area += la * lb
        print("inferred fault dimensions (km)", la, lb)
        points = points.reshape((p1.ny, p1.nx, 2)) * 1e3
        iy = p1.ny // 2
        ix = p1.nx // 2
        dx = min(dx, np.amax(np.abs(points[iy, ix, :] - points[iy, ix - 1, :])))
        dy = min(dy, np.amax(np.abs(abs(points[iy, ix, :] - points[iy - 1, ix, :]))))

    def next_odd_integer(x):
        if (x == int(x)) & (int(x) % 2 == 1):
            return int(x)
        nearest_integer = int(x)
        # If nearest integer is even, add 1, else add 2
        next_odd = nearest_integer + (1 if nearest_integer % 2 == 0 else 2)
        return next_odd

    def get_fault_mesh_size(min_plane_area, total_area):
        if total_area < 40 * 80:
            return 500
        elif total_area > 100 * 200:
            return 1000
        else:
            return 700

    if mesh_size == "auto":
        mesh_size = get_fault_mesh_size(min_fault_plane_area, total_area)
    else:
        mesh_size = float(mesh_size)
    print(f"using a mesh size of {mesh_size}")
    print(min(dx, dy))
    inferred_spatial_zoom = next_odd_integer(min(dx, dy) / mesh_size)

    print(f"inferred spatial zoom {inferred_spatial_zoom}")
    is_static_solution = mfp.is_static_solution()

    return inferred_spatial_zoom, mesh_size, len(mfp.fault_planes), is_static_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="infer spatial zoom to match fault mesh size"
    )
    parser.add_argument("filename", help="filename of the srf file")
    parser.add_argument(
        "--fault_mesh_size",
        type=str,
        default="auto",
        help="""
        auto: inferred from fault dimensions
        else provide a value
        """,
    )
    parser.add_argument(
        "--proj",
        metavar="proj",
        help="proj4 string describing the projection",
        required=True,
    )
    args = parser.parse_args()
    inferred_spatial_zoom, mesh_size, n_segments, is_static_solution = infer_quantities(
        args.filename, args.proj, args.fault_mesh_size
    )
    print(inferred_spatial_zoom, mesh_size, n_segments, is_static_solution)
