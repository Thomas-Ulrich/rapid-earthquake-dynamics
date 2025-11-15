#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import argparse


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "gen-fl33-inputs",
        help=(
            "Generate yaml and netcdf input to be used with friction law 33/34 based "
            "on a kinematic model file"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("filename", help="kinematic model filename")

    parser.add_argument(
        "--interpolation_method",
        metavar="interpolation_method",
        default="linear",
        help="interpolation method",
        choices=["linear", "nearest", "slinear", "cubic", "quintic"],
    )
    parser.add_argument(
        "--proj",
        metavar=("proj"),
        help=("proj4 string describing the projection"),
        required=True,
    )
    parser.add_argument(
        "--PSRthreshold",
        help="peak slip rate threshold (0-1) to determine STF onset time and duration.",
        metavar="PSRthreshold",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--spatial_zoom",
        metavar="spatial_zoom",
        required=True,
        help="level of spatial upsampling",
        type=int,
    )
    parser.add_argument(
        "--write_paraview",
        dest="write_paraview",
        action="store_true",
        help="write also netcdf readable by paraview",
        default=False,
    )

    def run(args):
        from kinematic_models.generate_FL33_input_files import main

        main(args)

    parser.set_defaults(func=run)
