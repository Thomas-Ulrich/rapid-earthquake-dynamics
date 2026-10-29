#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024-2025 Thomas Ulrich

"""
Compute and export the critical nucleation area (area_crit) to an XDMF file
for visualization (e.g., in ParaView) using seissolxdmfwriter by importing
functions from dynworkflow.estimate_nucleation_radius.
"""

import argparse
import seissolxdmfwriter as sxw
from dynworkflow.estimate_nucleation_radius import (
    SeissolxdmfExtended,
    compute_G_and_CG,
    compute_area_crit,
    compute_fmin_interp,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export area_crit to XDMF for visualization."
    )
    parser.add_argument("faultfile", help="SeisSol xdmf fault output")
    parser.add_argument("faultyaml", help="YAML file describing fault parameters")
    parser.add_argument(
        "--materialyaml",
        default="yaml_files/material.yaml",
        help="YAML file describing Lame parameters",
    )
    parser.add_argument(
        "--output",
        default="area_crit.xdmf",
        help="Output XDMF filename",
    )
    args = parser.parse_args()

    print(f"Reading fault geometry from {args.faultfile}...")
    sx = SeissolxdmfExtended(args.faultfile)
    centers = sx.ComputeCellCenters()
    normals = sx.ComputeCellNormals()
    tags = sx.ReadFaultTags()

    print(f"Evaluating material model from {args.materialyaml}...")
    G, CG = compute_G_and_CG(centers, normals, args.materialyaml)

    print(f"Evaluating fault model from {args.faultyaml} and computing area_crit...")
    fmin_interpf = compute_fmin_interp()
    area_crit, n_excluded = compute_area_crit(
        centers, tags, G, CG, args.faultyaml, fmin_interpf
    )
    print(
        f"Finished computing area_crit ({n_excluded} cells "
        "excluded with strength_drop <= 0)."
    )

    print(f"Writing output to {args.output}...")
    sxw.write(
        args.output,
        sx.xyz,
        sx.connect,
        {
            "area_crit": area_crit,
            "fault-tag": tags,
        },
        {0.0: 0},
        reduce_precision=True,
    )
    print("Done! You can now open the output XDMF file in ParaView.")
