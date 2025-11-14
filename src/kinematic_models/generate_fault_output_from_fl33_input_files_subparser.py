#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import argparse


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "gen-fault-output",
        help="Generate a fault output from FL33 input files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fault_filename", help="fault.xdmf filename")
    parser.add_argument("yaml_filename", help="fault easi/yaml filename")

    parser.add_argument(
        "--output_file",
        help="path and prefix of the output file",
        default="fault_from_fl33_input",
    )
    parser.add_argument(
        "--stf",
        type=str,
        choices=["Yoffe", "Gaussian", "AsymmetricCosine"],
        default="Gaussian",
        help="the source time function to use",
    )
    parser.add_argument(
        "--dt",
        metavar="dt",
        default=0.5,
        help="sampling time of the output file",
        type=float,
    )

    def run(args):
        from kinematic_models.generate_fault_output_from_fl33_input_files import main

        main(args)

    parser.set_defaults(func=run)
