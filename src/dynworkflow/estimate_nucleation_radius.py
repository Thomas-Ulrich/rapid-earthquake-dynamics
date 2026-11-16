#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

"""
Critical nucleation radius around a given hypocenter, for a list of fault yaml files.

For each fault yaml, the nucleation radius R is the smallest radius around the
hypocenter such that

    sum_{cells with center within R} face_area / area_crit  >=  factor

where area_crit = max(Uenishi 2009 estimate, Galis et al. 2015 estimate).
R is limited to min(sqrt(0.15 * slip_area / pi), fault_width / 2).
R is searched on a grid of radii (default 500 m, 750 m, 1000 m, ... in steps of
250 m). If the criterion is not reached up to the
limit, the largest radius of the grid below the limit is returned.
"""

import argparse
import multiprocessing
import os
import time

import easi
import numpy as np
import seissolxdmf
from scipy import interpolate
from scipy.special import ellipe, ellipk


class SeissolxdmfExtended(seissolxdmf.seissolxdmf):
    def __init__(self, xdmfFilename):
        super().__init__(xdmfFilename)
        self.xyz = self.ReadGeometry()
        self.connect = self.ReadConnect()

    def ReadFaultTags(self):
        """Read fault tag array"""
        return self.Read1dData("fault-tag", self.nElements, isInt=True).T

    def _cross(self):
        return np.cross(
            self.xyz[self.connect[:, 1], :] - self.xyz[self.connect[:, 0], :],
            self.xyz[self.connect[:, 2], :] - self.xyz[self.connect[:, 0], :],
        )

    def ComputeCellCenters(self):
        """compute cell center array"""
        return (
            self.xyz[self.connect[:, 0]]
            + self.xyz[self.connect[:, 1]]
            + self.xyz[self.connect[:, 2]]
        ) / 3.0

    def ComputeCellNormals(self):
        """compute unit cell normals"""
        cross = self._cross()
        return cross / np.linalg.norm(cross, axis=1)[:, None]

    def ComputeCellAreas(self):
        """compute area of each cell"""
        return 0.5 * np.linalg.norm(self._cross(), axis=1)


def C_Uenishi(nu):
    "C(nu) in Uenishi 2009, see Galis 2015, eqs 19-21 (exact, vectorized)"
    nu = np.clip(nu, 0.0, 0.499)
    m = nu * (2.0 - nu)  # a^2, parameter of the complete elliptic integrals
    return (ellipe(m) + (1.0 - nu) * ellipk(m)) / (2.0 - nu)


def stiffness_average(G1, G2):
    return 2.0 * G1 * G2 / (G1 + G2)


def compute_G_and_CG(centers, normals, mat_yaml):
    "G and C_Uenishi(nu)*G across the fault (harmonic average of both sides)"
    nx = centers.shape[0]
    regions = np.ones((2 * nx,), dtype=int)
    centers_mp = np.vstack((centers + 0.1 * normals, centers - 0.1 * normals))
    out = easi.evaluate_model(centers_mp, regions, ["lambda", "mu"], mat_yaml)
    lambda_x = stiffness_average(out["lambda"][:nx], out["lambda"][nx:])
    G = stiffness_average(out["mu"][:nx], out["mu"][nx:])
    nu = 0.5 * lambda_x / (lambda_x + G)
    return G, C_Uenishi(nu) * G


def compute_slip_area(centers, tags, face_area, slip_yaml):
    "area of the fault where slip exceeds 5% of the max slip"
    slip = easi.evaluate_model(centers, tags, ["fault_slip"], slip_yaml)["fault_slip"]
    return np.sum(face_area[slip > 0.05 * np.amax(slip)])


def estimate_fault_width(centers):
    "smallest extent of the fault along its two principal directions"
    # PCA done with numpy (same result as sklearn PCA(n_components=2)), to avoid
    # importing sklearn (and pyarrow), which can fail in spawned workers
    centered = centers - centers.mean(axis=0)
    _, eigvecs = np.linalg.eigh(np.cov(centered.T))  # eigenvalues in ascending order
    points = centered @ eigvecs[:, ::-1][:, :2]  # two principal directions
    la, lb = np.amax(points, axis=0) - np.amin(points, axis=0)
    return min(la, lb)


def _f(x, S):
    # Galis et al. eq. (25)
    return np.sqrt(x) * (1.0 + S * (1.0 - np.sqrt(1.0 - 1.0 / x**2)))


def compute_fmin_interp():
    Si = np.logspace(-2, 2, num=500)
    # I verified that for S=100, fmin ~15 < 20
    x = np.linspace(1.01, 20, 1000)
    fmin = np.array([np.amin(_f(x, S)) for S in Si])
    return interpolate.interp1d(Si, fmin)


def compute_area_crit(centers, tags, G, CG, fault_yaml, fmin_interpf):
    """Critical nucleation area per cell (inf where not defined)."""
    out = easi.evaluate_model(
        centers,
        tags,
        ["d_c", "static_strength", "dynamic_strength", "tau_0"],
        fault_yaml,
    )
    Dc = out["d_c"]
    static_strength = out["static_strength"]
    dynamic_strength = out["dynamic_strength"]
    tau_0 = out["tau_0"]

    if not np.all(np.isfinite(Dc)) or np.any(Dc <= 0):
        raise ValueError(
            f"{fault_yaml}: d_c non-finite or <= 0 in "
            f"{np.count_nonzero(~(np.isfinite(Dc) & (Dc > 0)))} cells "
            f"(min d_c = {np.nanmin(Dc):.3g}): check the fault yaml file"
        )
    if not np.all(np.isfinite(CG)) or np.any(CG <= 0):
        raise ValueError(
            f"non-finite or <= 0 values in C(nu)*G (min = {np.nanmin(CG):.3g}): "
            "check the material yaml file"
        )

    strength_drop = static_strength - dynamic_strength
    eps = 0.001
    valid = strength_drop > 0

    stress_drop = np.maximum(eps, tau_0 - dynamic_strength)
    S = (static_strength - tau_0) / stress_drop
    # 0.01 - 100 is the range of numerical evaluation of the function
    fmin = fmin_interpf(np.clip(S, 0.01 + eps, 100 - eps))

    # cells with strength_drop <= 0 are set to inf below, silence the warning
    with np.errstate(divide="ignore"):
        L = 0.624 * CG / (strength_drop / Dc)
    area_uenishi = np.pi * L**2
    area_galis = (
        np.pi**3
        * strength_drop**2
        * G**2
        * Dc**2
        / (16.0 * fmin**4 * stress_drop**4)
    )
    area_crit = np.maximum(area_uenishi, area_galis)
    area_crit[~valid] = np.inf
    return area_crit, np.count_nonzero(~valid)


def compute_critical_nucleation_one_file(
    centers,
    dist,
    face_area,
    G,
    CG,
    tags,
    fault_yaml,
    max_radius,
    factor,
    radius_step,
    min_radius,
):
    """Critical radius for one fault yaml. centers etc. are already cropped to
    the sphere of radius max_radius around the hypocenter, dist = distance of
    each cell center to the hypocenter."""
    start_time = time.time()
    bn_fault_yaml = os.path.basename(fault_yaml)
    area_crit, n_excluded = compute_area_crit(
        centers, tags, G, CG, fault_yaml, compute_fmin_interp()
    )
    easi_runtime = time.time() - start_time

    order = np.argsort(dist)
    cumsum = np.cumsum((face_area / area_crit)[order])
    k = np.searchsorted(cumsum, factor)  # first cell where the sum reaches factor
    # candidate radii: min_radius, min_radius + radius_step, ... <= max_radius
    # (as in the original script); the result is the first one satisfying the criterion
    reached = False
    if k < len(cumsum):
        nuc_radius = max(
            min_radius, np.ceil(dist[order][k] / radius_step) * radius_step
        )
        reached = nuc_radius <= max_radius
    if not reached:
        # we still try with the largest candidate radius, even if it may not nucleate
        if max_radius >= min_radius:
            nuc_radius = (
                min_radius
                + np.floor((max_radius - min_radius) / radius_step) * radius_step
            )
        else:
            nuc_radius = max_radius

    runtime = time.time() - start_time
    return {
        "yaml": bn_fault_yaml,
        "radius": nuc_radius,
        "reached": reached,
        "n_excluded": n_excluded,
        "n_cells": len(dist),
        "runtime": runtime,
        "easi_runtime": easi_runtime,
    }


def _worker(indexed_args):
    i, args = indexed_args
    return i, compute_critical_nucleation_one_file(*args)


def compute_critical_nucleation(
    fault_xdmf,
    mat_yaml,
    slip_yaml,
    list_fault_yaml,
    hypo_coords,
    factor=4.0,
    max_radius=None,
    nproc=None,
    radius_step=250.0,
    min_radius=500.0,
):
    """Returns the list of critical nucleation radii, one per fault yaml, in the
    same order. Radii are on the grid min_radius + n * radius_step; if the
    criterion is not reached, the largest radius of the grid below the max
    radius is returned."""
    sx = SeissolxdmfExtended(fault_xdmf)
    centers = sx.ComputeCellCenters()
    face_area = sx.ComputeCellAreas()
    tags = sx.ReadFaultTags()

    width = estimate_fault_width(centers)
    slip_area = compute_slip_area(centers, tags, face_area, slip_yaml)
    limit = min(np.sqrt((0.15 / np.pi) * slip_area), 0.5 * width)
    print(f"estimated fault width: {width:.1f}")
    print(
        f"slip area: {slip_area:.4g} ->"
        f"radius limit (15% slip area, width/2): {limit:.1f}"
    )
    if max_radius is not None:
        limit = max_radius
        print(f"using user-defined max radius: {limit:.1f}")

    hypo = np.array(hypo_coords, dtype=float)
    dist_all = np.linalg.norm(centers - hypo, axis=1)
    print(f"closest fault cell center to the hypocenter: {dist_all.min():.1f}")
    ids = dist_all <= limit
    if not np.any(ids):
        raise ValueError(
            f"no fault cell within {limit:.1f} of the hypocenter {hypo}: "
            "check the hypocenter coordinates"
        )

    normals = sx.ComputeCellNormals()[ids]
    centers, dist, face_area, tags = (
        centers[ids],
        dist_all[ids],
        face_area[ids],
        tags[ids],
    )
    G, CG = compute_G_and_CG(centers, normals, mat_yaml)

    args_list = [
        (
            centers,
            dist,
            face_area,
            G,
            CG,
            tags,
            fault_yaml,
            limit,
            factor,
            radius_step,
            min_radius,
        )
        for fault_yaml in list_fault_yaml
    ]

    if nproc is None:
        omp = os.environ.get("OMP_NUM_THREADS")
        nproc = int(omp) if omp else multiprocessing.cpu_count()
    nproc = max(1, min(nproc, len(args_list)))

    def report(r):
        "print the result of one yaml as soon as it is available"
        rad = r["radius"]
        ratio = 100 * np.pi * rad**2 / slip_area
        msg = f"{r['yaml']}: {rad:.0f} {ratio:.1f}"
        if not r["reached"]:
            msg += "  [criterion not reached: largest candidate radius used]"
        if r["n_excluded"]:
            msg += f"  [{r['n_excluded']} cells excluded: strength_drop<=0]"
        print(
            msg + f"  ({r['runtime']:.3f} s, easi {r['easi_runtime']:.3f} s)",
            flush=True,
        )

    print(f"fault_yaml: radius ratio_slip_area[%] (cells in sphere: {len(dist)})")
    results = [None] * len(args_list)
    indexed_args = list(enumerate(args_list))
    if nproc == 1:
        for ia in indexed_args:
            i, r = _worker(ia)
            results[i] = r
            report(r)
    else:
        print(f"using {nproc} processes", flush=True)
        # "spawn" instead of the default "fork": easi/ASAGI (OpenMP, threads) has
        # already been used in this process to evaluate the material, and forking
        # after that can deadlock the children
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=nproc) as pool:
            # unordered: each result is printed as soon as it is finished
            for i, r in pool.imap_unordered(_worker, indexed_args):
                results[i] = r
                report(r)
    # the returned list follows the order of list_fault_yaml
    return [float(r["radius"]) for r in results]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="compute the critical nucleation radius around a hypocenter "
        "for a list of fault yaml files"
    )
    parser.add_argument("faultfile", help="seissol xdmf fault output")
    parser.add_argument("hypocenter", type=float, nargs=3, help="hypocenter coords")
    parser.add_argument(
        "faultyamlfiles", nargs="+", help="yaml file(s) describing fault parameters"
    )
    parser.add_argument(
        "--materialyamlfile",
        help="yaml file describing Lame parameters",
        default="yaml_files/material.yaml",
    )
    parser.add_argument(
        "--slipyamlfile",
        help="yaml file describing fault slip",
        default="yaml_files/fault_slip.yaml",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=4.0,
        help="required sum(face_area/area_crit) (1.0 is the theoretical minimum)",
    )
    parser.add_argument(
        "--max-radius",
        type=float,
        default=None,
        help="override the max radius (default: min(sqrt(0.15*slip_area/pi), width/2))",
    )
    parser.add_argument(
        "--radius-step",
        type=float,
        default=250.0,
        help="step of the candidate radii, in mesh units (default: 250)",
    )
    parser.add_argument(
        "--min-radius",
        type=float,
        default=500.0,
        help="smallest candidate radius, in mesh units (default: 500)",
    )
    parser.add_argument("--nproc", type=int, default=None, help="number of processes")
    args = parser.parse_args()

    compute_critical_nucleation(
        args.faultfile,
        args.materialyamlfile,
        args.slipyamlfile,
        args.faultyamlfiles,
        args.hypocenter,
        factor=args.factor,
        max_radius=args.max_radius,
        nproc=args.nproc,
        radius_step=args.radius_step,
        min_radius=args.min_radius,
    )
