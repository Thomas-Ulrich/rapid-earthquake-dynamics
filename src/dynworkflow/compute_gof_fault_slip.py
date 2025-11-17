#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import glob
import os

import numpy as np
import pandas as pd
import seissolxdmf as sx
import tqdm


def fuzzysort(arr, idx, dim=0, tol=1e-6):
    """
    Return indexes of sorted points robust to small perturbations of individual
    components.
    Reference: https://stackoverflow.com/questions/19072110/
    numpy-np-lexsort-with-fuzzy-tolerant-comparisons
    Note: I added dim < arr.shape[0] - 1 in some if statement
    (else it will crash sometimes).
    """
    arrd = arr[dim]
    srtdidx = sorted(idx, key=arrd.__getitem__)

    i, ix = 0, srtdidx[0]
    for j, jx in enumerate(srtdidx[1:], start=1):
        if arrd[jx] - arrd[ix] >= tol:
            if j - i > 1 and dim < arr.shape[0] - 1:
                srtdidx[i:j] = fuzzysort(arr, srtdidx[i:j], dim + 1, tol)
            i, ix = j, jx

    if i != j and dim < arr.shape[0] - 1:
        srtdidx[i:] = fuzzysort(arr, srtdidx[i:], dim + 1, tol)

    return srtdidx


def lookup_sorted_geom(geom, atol):
    """return the indices to sort the
    geometry array by x, then y, then z
    and the associated inverse look-up table
    """
    ind = fuzzysort(geom.T, list(range(0, geom.shape[0])), tol=atol)
    # generate inverse look-up table
    dic = {i: index for i, index in enumerate(ind)}
    ind_inv = np.zeros_like(ind)
    for k, v in dic.items():
        ind_inv[v] = k
    return ind, ind_inv


# These 2 latter modules are on pypi (e.g. pip install seissolxdmf)
class seissolxdmfExtended(sx.seissolxdmf):
    def __init__(self, xdmfFilename, atol):
        super().__init__(xdmfFilename)
        self.geometry = self.ReadGeometry()
        self.connect = self.ReadConnect()
        self.sort(atol)
        self.asl = self.ReadData("ASl", self.ndt - 1)

    def ReadTimeStep(self):
        try:
            return super().ReadTimeStep()
        except NameError:
            return 0.0

    def compute_areas(self):
        triangles = self.geometry[self.connect, :]
        a = triangles[:, 1, :] - triangles[:, 0, :]
        b = triangles[:, 2, :] - triangles[:, 0, :]
        return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)

    def sort(self, atol):
        """sort geom array and reindex connect array to match the new geom array"""
        import trimesh

        trimesh.tol.merge = atol
        mesh = trimesh.Trimesh(self.geometry, self.connect)
        mesh.merge_vertices()
        self.geometry = mesh.vertices
        self.connect = mesh.faces

        ind, ind_inv = lookup_sorted_geom(self.geometry, atol)
        self.geometry = self.geometry[ind, :]
        connect = self.connect
        connect = np.array([ind_inv[x] for x in connect.flatten()]).reshape(
            connect.shape
        )
        # sort along line (then we can use multidim_intersect)
        self.connect = np.sort(connect, axis=1)


def multidim_intersect(arr1, arr2):
    """find indexes of same triangles in 2 connect arrays
    (associated with the same geom array)
    generate 1D arrays of tuples and use numpy function
    https://stackoverflow.com/questions/9269681/intersection-of-2d-numpy-ndarrays
    """
    arr1_view = arr1.view([("", arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([("", arr2.dtype)] * arr2.shape[1])
    intersected, ind1, ind2 = np.intersect1d(arr1_view, arr2_view, return_indices=True)
    return ind1, ind2


def area_weighted_ssq(areas, q):
    """Return area-weighted sum of squared values (∑ A_i q_i²)."""
    return np.dot(areas, q**2)


def compute_gof_fault_slip(output_folder, reference_model, atol=1e-3):
    if os.path.exists(output_folder):
        output_folder += "/"
    fault_output_files = sorted(glob.glob(f"{output_folder}*-fault.xdmf"))
    """
    fault_output_files = [
        fn for fn in fault_output_files if "dyn-kinmod" not in fn and "fl33" not in fn
    ]
    """
    sx_ref = seissolxdmfExtended(reference_model, atol)
    slip_threshold = 0.05
    print(f"using slip threshold of {slip_threshold}m")
    id_pos = sx_ref.asl > slip_threshold
    areas = sx_ref.compute_areas()
    areas_pos = areas[id_pos]
    total_area = areas_pos.sum()
    ref_rms = np.sqrt(np.dot(areas_pos, sx_ref.asl[id_pos] ** 2) / total_area)

    results = {
        "faultfn": [],
        "gof_slip": [],
    }

    for fo in tqdm.tqdm(fault_output_files):
        sx = seissolxdmfExtended(fo)
        if sx_ref.geometry.shape[0] != sx.geometry.shape[0]:
            n_ref = sx_ref.geometry.shape[0]
            n_cur = sx.geometry.shape[0]
            raise ValueError(
                "Mesh node mismatch:\n"
                f"  Reference model ({reference_model}): {n_ref} nodes\n"
                f"  Current model ({fo}): {n_cur} nodes\n"
                "Meshes must have the same number of nodes to compare."
            )
        ind1, ind2 = multidim_intersect(sx_ref.connect, sx.connect)
        id_pos = sx_ref.asl[ind1] > slip_threshold
        areas_pos = areas[ind1][id_pos]

        misfit = (
            area_weighted_ssq(areas_pos, (sx_ref.asl[ind1] - sx.asl[ind2])[id_pos])
            / total_area
        )
        misfit = np.sqrt(misfit)
        gof = np.exp(-misfit / ref_rms)
        results["faultfn"].append(fo)
        results["gof_slip"].append(gof)
    df = pd.DataFrame(results)
    pd.set_option("display.max_colwidth", None)
    print(df)
    fname = "gof_slip.pkl"
    df.to_pickle(fname)


def main(args):
    atol = args.atol[0]
    print(
        (
            "computing the gof to the reference fault slip distribution"
            f" {args.reference_model}..."
        )
    )
    compute_gof_fault_slip(args.output_folder, args.reference_model, atol)
