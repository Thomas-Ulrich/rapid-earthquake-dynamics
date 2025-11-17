#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import argparse

import argcomplete

from kinematic_models.compute_moment_rate_function_subparser import (
    add_parser as comp_mrf_add_parser,
)
from kinematic_models.generate_fault_output_from_fl33_input_files_subparser import (
    add_parser as gen_fo_add_parser,
)
from kinematic_models.generate_FL33_input_files_subparser import (
    add_parser as gen_fl33_add_parser,
)
from kinematic_models.project_fault_tractions_onto_asagi_grid_subparser import (
    add_parser as proj_ft_add_parser,
)

# from refine_srf_subparser import add_parser as ref_srf_add_parser


def main():
    parser = argparse.ArgumentParser(prog="kimo", description="Kinematic models CLI")
    subparsers = parser.add_subparsers(title="subcommands", dest="command")
    subparsers.required = True

    # Register subcommands
    for add_sub in [
        comp_mrf_add_parser,
        gen_fo_add_parser,
        gen_fl33_add_parser,
        proj_ft_add_parser,
        # ref_srf_add_parser,
    ]:
        add_sub(subparsers)

    # Enable autocomplete
    argcomplete.autocomplete(parser)

    # Parse and dispatch
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
