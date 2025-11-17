#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import argparse


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "add-sources",
        help=(
            "Update waveform configuration YAMLs with source files for synthetic"
            " seismograms."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    def run(args):
        from dynworkflow.add_source_files_to_waveform_config import main

        main(args)

    parser.set_defaults(func=run)
    return parser
