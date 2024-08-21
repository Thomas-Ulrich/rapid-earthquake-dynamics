#!/usr/bin/env python3
import seissolxdmf
import seissolxdmfwriter as sxw
import numpy as np
import argparse
import os


def remove_unused_nodes(xyz, connect):
    # Step 1: Create a list of unique nodes present in the connectivity array
    unique_nodes = list(set(connect.flatten()))
    # Step 2: Create a new geometry array with only the vertices present in the mesh
    new_geometry = xyz[unique_nodes]
    # Step 3: Renumber the connectivity array based on the new vertex indices
    node_map = {node: i for i, node in enumerate(unique_nodes)}
    new_connectivity = np.array([[node_map[v] for v in tri] for tri in connect])
    return new_geometry, new_connectivity


def remove_duplicate_nodes(points, tolerance=1e-5):
    # Round the points to a certain precision to identify duplicates
    rounded_points = np.round(points / tolerance).astype(np.int64)

    # Find unique points and their corresponding indices
    _, unique_indices, inverse_indices = np.unique(
        rounded_points, axis=0, return_index=True, return_inverse=True
    )

    # Select the unique points
    unique_points = points[unique_indices]

    return unique_points, inverse_indices


def remove_duplicate_triangles(triangles, triangle_data):
    # Sort the vertices of each triangle to ensure consistent ordering
    sorted_triangles = np.sort(triangles, axis=1)

    # Find unique triangles
    _, unique_indices = np.unique(sorted_triangles, axis=0, return_index=True)
    # Select the unique triangles and their associated data
    unique_triangles = triangles[unique_indices]
    unique_triangle_data = triangle_data[unique_indices]

    return unique_triangles, unique_triangle_data


def remove_duplicate_nodes_and_triangles(xyz, connect, BC):
    xyz, inverse_indices = remove_duplicate_nodes(xyz)
    # Update triangles to use the new node indices
    connect = inverse_indices[connect]
    connect, BC = remove_duplicate_triangles(connect, BC)
    return xyz, connect, BC


def generate_boundary_file(filename, iBC):
    sx = seissolxdmf.seissolxdmf(filename)
    xyz = sx.ReadGeometry()
    tetra = sx.ReadConnect()
    nElements = np.shape(tetra)[0]
    boundary = sx.ReadDataChunk("boundary", firstElement=0, nchunk=nElements, idt=0)

    def get_tetras_with_passing_boundary_condition(boundary, faceId, BC):
        boundaryFace = (boundary >> (faceId * 8)) & 0xFF
        if BC == "all":
            return np.where(boundaryFace > 0)[0]
        elif BC == "faults":
            is_fault = np.invert(np.isin(boundaryFace, [0, 1, 5]))
            return np.where(is_fault)[0]
        else:
            return np.where(boundaryFace == BC)[0]

    def compute_output_mesh_size(boundary, iBC):
        NS = 0
        for faceId in range(0, 4):
            tetraId = get_tetras_with_passing_boundary_condition(boundary, faceId, iBC)
            NS = NS + len(tetraId)
        assert NS != 0
        return NS

    NS = compute_output_mesh_size(boundary, iBC)
    connect = np.zeros((NS, 3), dtype=int)
    BC = np.zeros((1, NS))

    s_vert = np.zeros((4, 3), int)
    s_vert[0, :] = [0, 2, 1]
    s_vert[1, :] = [0, 1, 3]
    s_vert[3, :] = [0, 3, 2]
    s_vert[2, :] = [1, 2, 3]

    currentindex = 0
    for faceId in range(0, 4):
        boundaryFace = (boundary >> (faceId * 8)) & 0xFF
        tetraId = get_tetras_with_passing_boundary_condition(boundary, faceId, iBC)
        for idBound in range(0, len(tetraId)):
            trinodes = tetra[tetraId[idBound], s_vert[faceId, :]]
            connect[currentindex, :] = trinodes
            BC[0, currentindex] = boundaryFace[tetraId[idBound]]
            currentindex = currentindex + 1
    print(np.unique(BC))
    xyz, connect = remove_unused_nodes(xyz, connect)
    xyz, connect, BC = remove_duplicate_nodes_and_triangles(xyz, connect, BC[0])
    prefix, ext = os.path.splitext(os.path.basename(filename))

    sBC = "BC" if iBC != "faults" else "fault-tag"
    sxw.write(
        f"{prefix}_bc_{iBC}",
        xyz,
        connect,
        {sBC: BC},
        {},
        reduce_precision=True,
        backend="hdf5",
    )


def custom_choice(value):
    # Allow "all" and "fault" directly
    allowed_strings = ["all", "free-surface", "absorbing", "faults"]
    if value == "free-surface":
        return 1
    elif value == "absorbing":
        return 5
    elif value in allowed_strings:
        return value
    # Try to convert the value to an integer
    try:
        ivalue = int(value)
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid choice: '{value}' (choose from {allowed_strings}, or an integer)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read a PUML mesh and create a xdmf/h5 file containing the surface boundary mesh"
    )
    parser.add_argument(
        "filename",
        help="PUML mesh",
    )
    parser.add_argument(
        "BC",
        type=custom_choice,
        help="boundary condition can be 'all', 'free-surface', 'faults', 'absorbing' or an integer.",
    )

    args = parser.parse_args()

    generate_boundary_file(args.filename, args.BC)
