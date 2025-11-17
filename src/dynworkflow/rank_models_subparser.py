#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import argparse


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "rank-models",
        help="Gather computed metrics and rank models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "output_folder",
        help="path to output folder or full path to specific energy.csv file",
    )
    parser.add_argument(
        "--extension",
        help="figure extension (without the .)",
        default="pdf",
    )
    parser.add_argument("--font_size", help="font size", nargs=1, default=[8], type=int)

    parser.add_argument(
        "--gof_threshold",
        help="gof threshold from which results are selected",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--nmin",
        help="minimum number of synthetic moment rates drawn in color",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--nmax",
        help="maximum number of synthetic moment rates drawn in color",
        type=int,
        default=10,
    )

    def run(args):
        from dynworkflow.rank_models import main

        main(args)

    parser.set_defaults(func=run)
    return parser
