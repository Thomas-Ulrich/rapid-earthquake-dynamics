#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import gmsh
import numpy as np
import glob
import re
import argparse
import os
import pyvista as pv
import vtk


def generate(h_domain, h_fault, interactive, vertex_union_tolerance):
    # domain dimensions
    length_added = 200e3
    z0, z1 = -length_added, 0
    # z0, z1 = -length_added, 5e3

    gmsh.initialize()

    gmsh.model.add("finite-fault")

    # We can log all messages for further processing with:
    gmsh.logger.start()

    # Regular expression patterns to match vertex lines
    vertex_pattern = re.compile(r"VRTX (\d+) ([\d.-]+) ([\d.-]+) ([\d.-]+)")

    # Dictionary to store all unique vertices: {(x,y,z): gmsh_point_id}
    vertex_dict = {}
    # List to keep track of all vertices for later use
    allv = []
    faults = []

    # Function to find nearest point or add a new one
    def get_point_id(x, y, z, tolerance=vertex_union_tolerance):
        # Check if a nearby point exists
        for (ex, ey, ez), point_id in vertex_dict.items():
            dist2 = (x - ex) ** 2 + (y - ey) ** 2 + (z - ez) ** 2
            if dist2 < tolerance**2:
                print(
                    f"Found an almost duplicated vertex at ({x},{y},{z}), "
                    f"merging with ({ex},{ey},{ez})"
                )
                return point_id, (ex, ey, ez)

        # If no duplicate found, create a new point
        new_point_id = gmsh.model.occ.addPoint(x, y, z)
        vertex_dict[(x, y, z)] = new_point_id
        return new_point_id, (x, y, z)

    ts_files = sorted(glob.glob("tmp/*.ts"))
    print("generating mesh based on the following ts_files", ts_files)

    fault_meshes = []
    faces = np.hstack([[3, 0, 1, 2], [3, 0, 2, 3]])

    for i, fn in enumerate(ts_files):
        file_vertices = []
        with open(fn, "r") as file:
            for line in file:
                match = vertex_pattern.match(line)
                if match:
                    vertex_id, x, y, z = map(float, match.groups())
                    if z > -h_fault:
                        z = 0.0

                    # Get or create the point ID and get the actual coordinates used
                    point_id, coords = get_point_id(x, y, z)
                    file_vertices.append(coords)

        fault_mesh = pv.PolyData(file_vertices, faces)
        fault_meshes.append(fault_mesh)

        # Add unique vertices to allv
        allv.extend(file_vertices)

        vertices = np.array(file_vertices)

        if len(vertices) >= 4:  # Assuming we need at least 4 points for a surface
            # Get the Gmsh point IDs for our vertices
            point1 = vertex_dict[tuple(vertices[0])]
            point2 = vertex_dict[tuple(vertices[1])]
            point3 = vertex_dict[tuple(vertices[2])]
            point4 = vertex_dict[tuple(vertices[3])]

            # Create lines for triangle 1: (point1, point2, point3)
            line1 = gmsh.model.occ.addLine(point1, point2)
            line2 = gmsh.model.occ.addLine(point2, point3)
            line3 = gmsh.model.occ.addLine(point3, point1)
            curve_loop1 = gmsh.model.occ.addCurveLoop([line1, line2, line3])
            triangle1 = gmsh.model.occ.addPlaneSurface([curve_loop1])

            # Create lines for triangle 2: (point1, point3, point4)
            # Reuse line3 (point3 -> point1) in reverse
            line4 = gmsh.model.occ.addLine(point3, point4)
            line5 = gmsh.model.occ.addLine(point4, point1)
            curve_loop2 = gmsh.model.occ.addCurveLoop([line4, line5, -line3])
            triangle2 = gmsh.model.occ.addPlaneSurface([curve_loop2])

            # # Store both triangles as the fault
            faults.append((2, triangle1))
            faults.append((2, triangle2))

    # compute fault normals
    gmsh.model.occ.synchronize()

    fault_normals = []
    for i, fault in enumerate(faults):
        fault_normal = gmsh.model.getNormal(fault[1], [0.5, 0.5])
        fault_normals.append(fault_normal)

    # Convert allv to numpy array for further processing
    allv = np.array(allv)
    min_x, max_x = np.min(allv[:, 0]), np.max(allv[:, 0])
    min_y, max_y = np.min(allv[:, 1]), np.max(allv[:, 1])

    x0 = min_x - length_added
    x1 = max_x + length_added
    y0 = min_y - length_added
    y1 = max_y + length_added
    box = gmsh.model.occ.addBox(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)

    gmsh.model.occ.synchronize()
    ov, ovv = gmsh.model.occ.fragment([(3, box)], faults)
    gmsh.model.occ.synchronize()

    # now retrieves which surface is where
    all_points = gmsh.model.getEntities(dim=0)
    coords = np.zeros((len(all_points), 3))
    pt2vertexid = {}
    for i, pt in enumerate(all_points):
        coords[i, :] = gmsh.model.getValue(*pt, [])
        pt2vertexid[pt[1]] = i
    print(coords)

    all_surfaces = gmsh.model.getEntities(dim=2)

    points_of_surface = {}
    tags = {}
    for i in [1, 5]:
        tags[i] = []
    for i, fn in enumerate(ts_files):
        fault_id = 3 if i == 0 else 64 + i
        tags[fault_id] = []

    for surface in all_surfaces:
        curves = gmsh.model.getBoundary([surface])
        pts = set()
        for cu in curves:
            points = gmsh.model.getBoundary([cu])
            points = [pt[1] for pt in points]
            pts.update(points)
        points_of_surface[surface] = pts
        vids = [pt2vertexid[pt] for pt in pts]
        surf_coords = coords[list(vids), :]
        zmin = np.min(surf_coords[:, 2])
        stag = surface[1]
        # print('s',tag, pts)
        if zmin > -0.01:
            # free surface
            tags[1].append(stag)
        elif abs(zmin - z0) < 0.01:
            # absorbing
            tags[5].append(stag)
        else:
            tagged = False
            for i, fault_mesh in enumerate(fault_meshes):
                fault_id = 3 if i == 0 else 64 + i

                ipd = vtk.vtkImplicitPolyDataDistance()
                ipd.SetInput(fault_mesh)

                distances = np.array([ipd.EvaluateFunction(pt) for pt in surf_coords])
                if np.all(distances < vertex_union_tolerance):
                    tags[fault_id].append(stag)
                    tagged = True
                    break
            if not tagged:
                raise ValueError(f"surface {stag} could not be tagged, {surf_coords}")
    print(tags)

    for key in tags.keys():
        h = h_domain if key in [1, 5] else h_fault
        pairs = [(2, tag) for tag in tags[key]]
        gmsh.model.mesh.setSize(gmsh.model.getBoundary(pairs, False, False, True), h)
        gmsh.model.addPhysicalGroup(2, tags[key], key)

    fault_faces = []
    for key in tags.keys():
        if key not in [1, 5]:
            fault_faces.extend(tags[key])
    print(fault_faces)

    # Set mesh size based on a distance from faults field
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", fault_faces)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)
    gmsh.model.mesh.field.add("MathEval", 2)
    gmsh.model.mesh.field.setString(2, "F", f"20*F1^(0.5) + 0.1*F1 + {h_fault}")
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(3, [1], 1)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.model.mesh.generate(2)

    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    gmsh.write("tmp/mesh2d.stl")
    gmsh.model.mesh.generate(3)
    gmsh.write("tmp/mesh.msh")

    if interactive:
        gmsh.fltk.run()
    gmsh.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generate a mesh given fault data from a kinematic model"
    )

    parser.add_argument(
        "--domain_mesh_size",
        help="mesh size in the domain",
        type=float,
        default=20000,
    )
    parser.add_argument(
        "--fault_mesh_size",
        help="mesh size on the faults",
        type=float,
        default=1000,
    )

    parser.add_argument(
        "--vertex_union_tolerance",
        help="minimum distance below which vertices are merged",
        type=float,
        default=500,
    )

    parser.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        help="open the gui of gmsh once the mesh is generated",
        default=False,
    )

    args = parser.parse_args()
    h_domain = args.domain_mesh_size
    h_fault = args.fault_mesh_size
    generate(h_domain, h_fault, args.interactive, args.vertex_union_tolerance)
