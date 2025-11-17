#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich, Mathilde Marchandon

import argparse


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "fault-offsets",
        help="""Compute fit (RMS) to offset of models from an
        ensemble of DR models.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("output_folder", help="folder where the models lie")
    parser.add_argument("offset_data", help="path to offset data")
    parser.add_argument(
        "--bestmodel", type=str, help='Pattern for best model (e.g. "dyn_0073")'
    )

    parser.add_argument(
        "--threshold_z",
        help="threshold depth used for selecting fault trace nodes",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--individual_figures",
        action="store_true",
        help="plot one figure for each file",
    )

    def run(args):
        from dynworkflow.compare_offset import main

        main(args)

    parser.set_defaults(func=run)
    return parser
