#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import argparse


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "tractions-to-asagi",
        help=(
            "Project 3D fault tractions onto 2D netcdf grids that can be"
            " read by ASAGI. One grid per fault tag."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("fault_filename", help="Fault.xdmf filename")
    parser.add_argument(
        "--dx",
        help="Grid sampling",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--gaussian_kernel",
        metavar="sigma_m",
        help="Apply a Gaussian kernel to smooth out input stresses",
        type=float,
    )
    parser.add_argument(
        "--edge_clearance",
        metavar="n_samples",
        help="Nullify traction near the left, bottom, and right edges of the grid.",
        type=int,
    )

    parser.add_argument(
        "--use_median_of_n_time_steps",
        type=int,
        metavar="N",
        help=(
            "Use the median of the last N time steps instead of the final snapshot. "
            "This helps suppress transient effects in the data."
        ),
        default=7,
    )

    parser.add_argument(
        "--taper",
        help="Taper stress value (MPa)",
        type=float,
    )
    parser.add_argument(
        "--paraview_readable",
        dest="paraview_readable",
        action="store_true",
        help="Write NetCDF files readable by ParaView",
        default=False,
    )

    def run(args):
        from kinematic_models.project_fault_tractions_onto_asagi_grid import main

        main(args)

    parser.set_defaults(func=run)
