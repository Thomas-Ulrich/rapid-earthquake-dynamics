#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import argparse


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "gen-dr-inputs",
        help=(
            "Process pseudo-static simulation output and generate input files"
            " for the ensemble of dynamic rupture simulations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    def run(args):
        from dynworkflow.generate_input_files_for_dr_ensemble import main

        main(args)

    parser.set_defaults(func=run)
    return parser
