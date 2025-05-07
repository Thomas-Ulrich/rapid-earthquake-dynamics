#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import numpy as np
import argparse
from scipy import interpolate
import easi
import seissolxdmf
import os
import time
from multiprocessing import cpu_count


class SeissolxdmfExtended(seissolxdmf.seissolxdmf):
    def __init__(self, xdmfFilename):
        super().__init__(xdmfFilename)
        self.xyz = self.ReadGeometry()
        self.connect = self.ReadConnect()

    def ReadFaultTags(self):
        """Read partition array"""
        return self.Read1dData("fault-tag", self.nElements, isInt=True).T

    def ComputeCellCenters(self):
        """compute cell center array"""
        return (
            self.xyz[self.connect[:, 0]]
            + self.xyz[self.connect[:, 1]]
            + self.xyz[self.connect[:, 2]]
        ) / 3.0

    def ComputeCellNormals(self):
        """compute cell normal"""
        cross = np.cross(
            self.xyz[self.connect[:, 1], :] - self.xyz[self.connect[:, 0], :],
            self.xyz[self.connect[:, 2], :] - self.xyz[self.connect[:, 0], :],
        )
        norm = np.apply_along_axis(np.linalg.norm, 1, cross)
        return cross / norm.reshape((self.nElements, 1))

    def ComputeCellAreas(self):
        """compute area of each cell"""
        cross = np.cross(
            self.xyz[self.connect[:, 1], :] - self.xyz[self.connect[:, 0], :],
            self.xyz[self.connect[:, 2], :] - self.xyz[self.connect[:, 0], :],
        )
        return 0.5 * np.apply_along_axis(np.linalg.norm, 1, cross)


def k_int(x):
    "K integral in Uenishi 2009, see Galis 2015, eq 20"
    ti = np.linspace(0, 1, 250, endpoint=False)
    f = 1.0 / np.sqrt((1.0 - ti**2) * (1.0 - x**2 * ti**2))  # for t in ti]
    return np.trapz(f, ti)


def e_int(x):
    "E integral in Uenishi 2009, see Galis 2015, eq 21"
    ti = np.linspace(0, 1, 250, endpoint=False)
    f = [np.sqrt((1.0 - x**2 * t**2) / (1.0 - t**2)) for t in ti]
    return np.trapz(f, ti)


def C_Uenishi(x):
    "C(nu) in Uenishi 2009, see Galis 2015, eq 19"
    a = np.sqrt(x * (2.0 - x))
    return (e_int(a) + (1.0 - x) * k_int(a)) / (2.0 - x)


def C_Uenishi_interp():
    "interpolated function based on C_Uenishi (much faster)"
    x = np.arange(0.14, 0.35, 0.001)
    y = np.array([C_Uenishi(xi) for xi in x])
    return interpolate.interp1d(x, y)


def stiffness_average(G1, G2):
    return 2.0 * G1 * G2 / (G1 + G2)


def compute_G_and_CG(centers, ids, sx, mat_yaml):
    "compute nu across the fault, and compute C_Uenishi(nu)"
    fault_normal = sx.ComputeCellNormals()[ids]
    nx = fault_normal.shape[0]
    regions = np.ones((2 * nx,))
    centers_mp = np.vstack((centers + 0.1 * fault_normal, centers - 0.1 * fault_normal))
    out = easi.evaluate_model(centers_mp, regions, ["lambda", "mu"], mat_yaml)
    lambda_x = stiffness_average(out["lambda"][0:nx], out["lambda"][nx:])
    G = stiffness_average(out["mu"][0:nx], out["mu"][nx:])
    nu = 0.5 * lambda_x / (lambda_x + G)
    f = C_Uenishi_interp()
    return G, np.array([f(nui) for nui in nu]) * G


def points_in_sphere(points, center, radius):
    distances = np.linalg.norm(points - center, axis=1)
    within_sphere = distances <= radius
    return within_sphere


def compute_slip_area(centers, sx, slip_yaml):
    face_area = sx.ComputeCellAreas()
    tags = sx.ReadFaultTags()
    slip = easi.evaluate_model(centers, tags, ["fault_slip"], slip_yaml)["fault_slip"]
    return np.sum(face_area[slip > 0.05 * np.amax(slip)])


def f(x, S):
    # Galis et al. e.q. (25)
    return np.sqrt(x) * (1.0 + S * (1.0 - np.sqrt(1.0 - 1.0 / x**2)))


def compute_fmin_interp():
    Si = np.logspace(-2, 2, num=500)
    fmin = np.zeros_like(Si)
    # I verified that for S=100, fmin ~15 < 20
    x = np.linspace(1.01, 20, 1000)
    for i, S in enumerate(Si):
        fmin[i] = np.amin(f(x, S))
    return interpolate.interp1d(Si, fmin)


def compute_critical_nucleation_one_file(
    centers, center, face_area, slip_area, G, CG, fmin_interpf, tags, fault_yaml
):
    start_time = time.time()
    bn_fault_yaml = os.path.basename(fault_yaml)
    out = easi.evaluate_model(
        centers,
        tags,
        ["d_c", "static_strength", "dynamic_strength", "tau_0"],
        fault_yaml,
    )
    easi_runtime = time.time() - start_time

    Dc, static_strength, dynamic_strength, tau_0 = (
        out["d_c"],
        out["static_strength"],
        out["dynamic_strength"],
        out["tau_0"],
    )
    strength_drop = static_strength - dynamic_strength
    eps = 0.001
    S = (static_strength - tau_0) / np.maximum(eps, tau_0 - dynamic_strength)
    # 0.01- 100 is the range of numerical evaluation of the function
    fmin = fmin_interpf(np.maximum(np.minimum(S, 100 - eps), 0.01 + eps))

    W = strength_drop / Dc
    # L = 0.624 * CG / np.median(W)
    L = 0.624 * CG / W
    area_crit = np.pi * L**2

    A2_Gallis = (
        np.pi**3
        * strength_drop**2
        * G**2
        * Dc**2
        / (16.0 * fmin**4 * np.maximum(eps, tau_0 - dynamic_strength) ** 4)
    )
    area_crit = np.maximum(area_crit, A2_Gallis)
    L = np.sqrt(area_crit / np.pi)

    maxnucRadius = np.sqrt((0.15 / np.pi) * slip_area)
    radius = np.arange(0.5e3, maxnucRadius + 0.25e3, 0.25e3)

    nucRadius = False
    for k, rad in enumerate(radius):
        ratio_slip_area = 100 * np.pi * rad**2 / slip_area
        if ratio_slip_area > 15.0:
            nucRadius = radius[max(0, k - 1)]
            # min(rad, np.sqrt((0.15 / np.pi) * slip_area))
            ratio_slip_area = 100 * np.pi * nucRadius**2 / slip_area
            break

        ids = points_in_sphere(centers, center, rad)
        selected_area_ratio = face_area[ids] / area_crit[ids]
        sum_ratio = np.sum(selected_area_ratio)
        # 1.0 is what in theory needed
        if sum_ratio > 4.0:
            nucRadius = rad
            break
    runtime = time.time() - start_time
    print(
        f"{bn_fault_yaml}: {nucRadius:.0f} {ratio_slip_area:.1f} "
        f" ({runtime:.3f} s, easi {easi_runtime:.3f}s)"
    )
    return nucRadius


def compute_nucleation(args):
    centers, center, face_area, slip_area, G, CG, fmin_interpf, tags, fault_yaml = args
    return compute_critical_nucleation_one_file(
        centers, center, face_area, slip_area, G, CG, fmin_interpf, tags, fault_yaml
    )


def compute_critical_nucleation(
    fault_xdmf, mat_yaml, slip_yaml, list_fault_yaml, hypo_coords
):
    fmin_interpf = compute_fmin_interp()
    sx = SeissolxdmfExtended(fault_xdmf)
    centers = sx.ComputeCellCenters()
    slip_area = compute_slip_area(centers, sx, slip_yaml)

    center = np.array(hypo_coords)
    maxnucRadius = np.sqrt((0.15 / np.pi) * slip_area)
    ids = points_in_sphere(centers, center, maxnucRadius)
    centers = centers[ids]

    G, CG = compute_G_and_CG(centers, ids, sx, mat_yaml)
    tags = sx.ReadFaultTags()[ids]

    face_area = sx.ComputeCellAreas()[ids]

    args_list = [
        (centers, center, face_area, slip_area, G, CG, fmin_interpf, tags, fault_yaml)
        for fault_yaml in list_fault_yaml
    ]
    from multiprocessing import Pool

    def get_num_threads():
        omp_num_threads = os.environ.get("OMP_NUM_THREADS")
        if omp_num_threads:
            return int(omp_num_threads)  # Use the defined value
        return cpu_count()  # Fall back to the number of CPUs

    num_threads = get_num_threads()

    print(f"using {num_threads} threads")
    print("fault_yaml: selected_r ratio_slip_area")
    with Pool(processes=num_threads) as pool:
        async_result = pool.map_async(compute_nucleation, args_list)
        results = async_result.get()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute critical depletion")
    parser.add_argument(
        "faultfile", help="fault file (vtk or seissol xdmf fault output)"
    )
    parser.add_argument("faultyamlfile", help="yaml file describing fault parameters")
    parser.add_argument("hypocenter", type=float, nargs=3, help="hypocenter coords")
    parser.add_argument(
        "--materialyamlfile",
        help="yaml file desribing Lame parameters",
        default="yaml_files/material.yaml",
    )
    parser.add_argument(
        "--slipyamlfile",
        help="yaml file describing fault slip",
        default="yaml_files/fault_slip.yaml",
    )
    args = parser.parse_args()
    fault_xdmf = args.faultfile
    fault_yaml = args.faultyamlfile
    slip_yaml = args.slipyamlfile
    mat_yaml = args.materialyamlfile
    hypo = args.hypocenter

    compute_critical_nucleation(fault_xdmf, mat_yaml, slip_yaml, [fault_yaml], hypo)
