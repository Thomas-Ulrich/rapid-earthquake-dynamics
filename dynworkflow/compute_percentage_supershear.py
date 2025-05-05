#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import glob
import numpy as np
import argparse
import os
import seissolxdmf as sx
import pandas as pd
import tqdm
import easi


# These 2 latter modules are on pypi (e.g. pip install seissolxdmf)
class seissolxdmfExtended(sx.seissolxdmf):
    def __init__(self, xdmfFilename):
        super().__init__(xdmfFilename)
        self.geometry = self.ReadGeometry()
        self.connect = self.ReadConnect()
        self.vr = self.ReadData("Vr", self.ndt - 1)
        self.asl = self.ReadData("ASl", self.ndt - 1)
        self.xyzc = self.compute_cell_centers()

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

    def compute_cell_centers(self):
        return self.geometry[self.connect].mean(axis=1)

    def evaluate_vp_vs(self, material_file):
        regions = np.ones((self.nElements, 1))
        print("Warning: assuming region 1 for all cells when evaluating mu with easi")
        out = easi.evaluate_model(
            self.xyzc, regions, ["rho", "mu", "lambda"], material_file
        )
        self.vs = np.sqrt(out["mu"] / out["rho"])
        self.vp = np.sqrt((out["lambda"] + 2.0 * out["mu"]) / out["rho"])


def l2_norm(areas, q):
    return np.dot(areas, np.power(q, 2))


def compute_supershear_percentile(folder, material_file):
    if os.path.exists(args.output_folder):
        args.output_folder += "/"
    fault_output_files = sorted(glob.glob(f"{folder}*-fault.xdmf"))

    results = {
        "faultfn": [],
        "supershear": [],
    }

    for fo in tqdm.tqdm(fault_output_files):
        if "dyn-kinmod" in fo:
            supershear_percentile = np.nan
        else:
            sx = seissolxdmfExtended(fo)
            areas = sx.compute_areas()
            sx.evaluate_vp_vs(material_file)
            id_pos = sx.asl > 0.05
            # these 10% acknowledge the fact that
            # the supershear calculation can be imprecise
            supershear = sx.vr > sx.vs + 0.1 * (sx.vp - sx.vs)

            total_area = areas[id_pos].sum()
            supershear_area = areas[id_pos & supershear].sum()
            supershear_percentile = (supershear_area / total_area) * 100
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
