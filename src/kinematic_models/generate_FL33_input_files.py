#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import os
import os.path

from kinematic_models.multi_fault_plane import MultiFaultPlane


def generate(
    filename,
    interpolation_method,
    spatial_zoom,
    projection,
    write_paraview,
    PSRthreshold,
    tmax=None,
):
    prefix, ext = os.path.splitext(filename)
    prefix = os.path.basename(prefix)
    mfp = MultiFaultPlane.from_file(filename)
    if tmax:
        mfp.temporal_crop(tmax)

    for p, p1 in enumerate(mfp.fault_planes):
        p1.compute_time_array()
        if ext == ".srf":
            p1.assess_STF_parameters(PSRthreshold)
        p1.generate_netcdf_fl33(
            f"{prefix}{p + 1}",
            method=interpolation_method,
            spatial_zoom=spatial_zoom,
            proj=projection,
            write_paraview=write_paraview,
            slip_cutoff=0.0,
        )

    mfp.generate_fault_ts_yaml_fl33(
        prefix,
        method=interpolation_method,
        spatial_zoom=spatial_zoom,
        proj=projection,
    )


def main(args):
    generate(
        args.filename,
        args.interpolation_method,
        args.spatial_zoom,
        args.proj,
        args.write_paraview,
        args.PSRthreshold,
    )
