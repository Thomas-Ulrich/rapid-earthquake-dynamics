#!/usr/bin/env python3
import glob
import h5py
import numpy as np
import argparse
import os
import seissolxdmf as sx
import seissolxdmfwriter as sw
import pandas as pd
import tqdm


# These 2 latter modules are on pypi (e.g. pip install seissolxdmf)
class seissolxdmfExtended(sx.seissolxdmf):
    def __init__(self, xdmfFilename):
        super().__init__(xdmfFilename)
        self.geometry = self.ReadGeometry()
        self.connect = self.ReadConnect()
        self.vr = self.ReadData("Vr", self.ndt - 1)
        self.asl = self.ReadData("ASl", self.ndt - 1)
        self.depthz = -self.compute_cell_centers()[:, 2]

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


def l2_norm(areas, q):
    return np.dot(areas, np.power(q, 2))


def compute_supershear_percentile(folder, velocity_model):
    if os.path.exists(args.output_folder):
        args.output_folder += "/"
    fault_output_files = sorted(glob.glob(f"{folder}*-fault.xdmf"))

    results = {
        "faultfn": [],
        "supershear": [],
    }

    df = pd.read_csv(
        velocity_model,
        sep="\s+",
        comment="#",
        header=None,
        names=["layer_width", "Vp", "Vs", "rho", "Qp", "Qs"],
    )
    df.loc[df.index[-1], "layer_width"] = 1e10
    df["depth"] = df["layer_width"].cumsum()

    # Function to find the appropriate value in Mui for any z
    def find_last_value(zi, Vsi, z):
        idx = np.searchsorted(zi, z, side="right")
        return Vsi[idx]

    for fo in tqdm.tqdm(fault_output_files):
        if "dyn-kinmod" in fo:
            supershear_percentile = np.nan
        else:
            sx = seissolxdmfExtended(fo)
            areas = sx.compute_areas()
            id_pos = sx.asl > 0.05
            Vs = find_last_value(df["depth"], df["Vs"], sx.depthz)
            Vp = find_last_value(df["depth"], df["Vp"], sx.depthz)
            # these 10% acknowledge the fact that the supershear calculation can be imprecise
            supershear = sx.vr > Vs + 0.1 * (Vp - Vs)

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
    parser.add_argument(
        "velocity_model", help="axitra file describing the 1D velocity model"
    )

    args = parser.parse_args()
    compute_supershear_percentile(args.output_folder, args.velocity_model)
