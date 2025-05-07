#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import argparse
from multi_fault_plane import MultiFaultPlane


def main(
    finite_fault_filename,
    offset_filename,
):
    mfp = MultiFaultPlane.from_file(finite_fault_filename)
    for p, p1 in enumerate(mfp.fault_planes):
        p1.modify_shallow_slip_to_match_offset(offset_filename)
    mfp.write_param("modified.par")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("change shallow slip to match given offset")
    )
    parser.add_argument(
        "finite_fault_filename", help="filename of the finite-fault file"
    )
    parser.add_argument("offset_filename", help="filename of the offset csv file")
    args = parser.parse_args()
    main(
        args.finite_fault_filename,
        args.offset_filename,
    )
