#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import argparse


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "supershear",
        help="""compute percentage of supershear in slip area for an
        ensemble of DR models. all on the same mesh.
        partitionning may differ though""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("output_folder", help="folder where the models lie")
    parser.add_argument("material_file", help="easi yaml file defining rho mu lambda")

    def run(args):
        from dynworkflow.compute_percentage_supershear import main

        main(args)

    parser.set_defaults(func=run)
    return parser
