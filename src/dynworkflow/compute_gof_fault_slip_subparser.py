#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import argparse


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "slip",
        help="""compute fault slip difference between between models from an
        ensemble of DR models and a reference model, all on the same mesh.
        partitionning may differ though""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    def run(args):
        from dynworkflow.compute_gof_fault_slip import main

        main(args)

    parser.set_defaults(func=run)
    return parser
