#!/usr/bin/env python3
import glob
import h5py
import numpy as np
import argparse
import os
import seissolxdmf as sx
import seissolxdmfwriter as sw

# import generate_fault_output_from_fl33_input_files


def fuzzysort(arr, idx, dim=0, tol=1e-6):
    """
    return indexes of sorted points robust to small perturbations of individual components.
    https://stackoverflow.com/questions/19072110/numpy-np-lexsort-with-fuzzy-tolerant-comparisons
    note that I added dim<arr.shape[0]-1 in some if statement (else it will crash sometimes)
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
    def __init__(self, xdmfFilename):
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
    # ni, n1, n2 = intersected.shape[0], arr1.shape[0], arr2.shape[0]
    # print(
    #    f"{ni} faces in common, n faces connect 1:{n1}, 2:{n2} (diff: {n1-ni}, {n2-ni})"
    # )
    return ind1, ind2


def l1_norm(areas, q):
    return np.dot(areas, np.abs(q))


def l2_norm(areas, q):
    return np.dot(areas, np.power(q, 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""compute fault slip difference between between models from an 
        ensemble of DR models and a reference model, all on the same mesh.
        partitionning may differ though"""
    )
    parser.add_argument("output_folder", help="folder where the models lie")
    parser.add_argument("reference_model", help="path to reference model")
    parser.add_argument(
        "--atol",
        nargs=1,
        metavar=("atol"),
        help="absolute tolerance to merge vertices",
        type=float,
        default=[1e-3],
    )

    args = parser.parse_args()
    atol = args.atol[0]

    if os.path.exists(args.output_folder):
        args.output_folder += "/"
    fault_output_files = sorted(glob.glob(f"{args.output_folder}*-fault.xdmf"))

    sx_ref = seissolxdmfExtended(args.reference_model)
    max_slip = sx_ref.asl.max()
    areas = sx_ref.compute_areas()
    sum_areas = areas.sum()
    for fo in fault_output_files:
        sx = seissolxdmfExtended(fo)
        if sx_ref.geometry.shape[0] != sx.geometry.shape[0]:
            raise ValueError("meshes don't have the same number of nodes")
        ind1, ind2 = multidim_intersect(sx_ref.connect, sx.connect)
        misfit = l2_norm(areas, sx_ref.asl[ind1] - sx.asl[ind2]) / sum_areas
        misfit = np.sqrt(misfit)
        gof = max(0, 1 - misfit / max_slip)
        print(f"{fo} {misfit} {gof}")
