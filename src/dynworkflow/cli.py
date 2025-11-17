#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import argparse

import argcomplete

from dynworkflow.step1_args import add_parser as step1_add_parser
from dynworkflow.compute_gof_fault_slip_subparser import add_parser as slip_add_parser
from dynworkflow.rank_models_subparser import add_parser as rank_add_parser


def main():
    parser = argparse.ArgumentParser(
        prog="redyn", description="Rapid earthquake dynamics CLI"
    )
    subparsers = parser.add_subparsers(title="subcommands", dest="command")
    subparsers.required = True

    # Register subcommands
    for add_sub in [
        step1_add_parser,
    ]:
        add_sub(subparsers)

    # --- Add 'gof' subcommand ---
    gof_parser = subparsers.add_parser("metrics", help="Metrics commands")
    gof_subparsers = gof_parser.add_subparsers(
        title="metrics subcommands", dest="metrics_command"
    )
    gof_subparsers.required = True

    # Register subcommands for 'metrics'
    for add_sub in [
        slip_add_parser,
        rank_add_parser,
    ]:
        add_sub(gof_subparsers)

    # Enable autocomplete
    argcomplete.autocomplete(parser)

    # Parse and dispatch
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
