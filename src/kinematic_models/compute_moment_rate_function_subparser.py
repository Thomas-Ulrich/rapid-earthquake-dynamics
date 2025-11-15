#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import argparse


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "compute-mrf",
        help=(
            "Compute moment rate function from kinematic model. Assumptions: Gaussian "
            "source time function, no bimaterial conditions"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dt",
        metavar="dt",
        default=0.25,
        help="sampling time of the output file",
        type=float,
    )

    parser.add_argument(
        "filename",
        help="filename of the kinematic model (supported format include srf, param)",
    )
    parser.add_argument(
        "yaml_filename", help="material easi/yaml filename providing mu"
    )

    parser.add_argument(
        "--proj",
        metavar="proj",
        help="proj4 string describing the projection",
        required=True,
    )

    def run(args):
        from kinematic_models.compute_moment_rate_function import main

        main(args)

    parser.set_defaults(func=run)
