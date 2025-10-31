#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import argparse
import os

from multi_fault_plane import MultiFaultPlane


def refine(
    filename,
    mode,
    projection,
    yoffePSRthreshold,
    spatial_order,
    spatial_factor,
    temporal_factor,
    time_smoothing_kernel_as_dt_fraction,
):
    mfp = MultiFaultPlane.from_file(filename)
    # if tmax:
    #    mfp.temporal_crop(tmax)

    for p, p1 in enumerate(mfp.fault_planes):
        p1.compute_xy_from_latlon(projection)
        p1.compute_time_array()
        p1.init_aSR()

        use_Yoffe = True if yoffePSRthreshold else False
        if use_Yoffe:
            p1.assess_STF_parameters(yoffePSRthreshold)

        if mode == "upsample":
            p2 = p1.upsample_fault(
                spatial_order=spatial_order,
                spatial_zoom=spatial_factor,
                temporal_zoom=temporal_factor,
                proj=projection,
                use_Yoffe=use_Yoffe,
                time_smoothing_kernel_as_dt_fraction=time_smoothing_kernel_as_dt_fraction,
            )
        elif mode == "downsample":
            p2 = p1.downsample_fault(
                spatial_factor=spatial_factor,
                temporal_factor=temporal_factor,
            )
        else:
            raise ValueError("Mode must be 'upsample' or 'downsample'.")

        prefix, ext = os.path.splitext(filename)
        fnout = f"{prefix}_{mode}d.srf"
        p2.write_srf(fnout)
        print(f"{mode.capitalize()}d SRF written to {fnout}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Upsample temporally and spatially a kinematic model (should consist of "
            "only one segment) in the standard rupture format."
        )
    )
    parser.add_argument("filename", help="Filename of the SRF file.")
    parser.add_argument(
        "--mode",
        choices=["upsample", "downsample"],
        required=True,
        help="Choose whether to upsample or downsample the fault model.",
    )

    parser.add_argument(
        "--proj",
        help=(
            "Transform geometry given a proj4 string (as it might be better to upsample"
            " the geometry in the local coordinate system)."
        ),
    )
    parser.add_argument(
        "--spatial_order",
        nargs=1,
        metavar="spatial_order",
        default=[3],
        help="Spatial order of the interpolation.",
        type=int,
    )
    parser.add_argument(
        "--spatial_zoom",
        nargs=1,
        metavar="spatial_zoom",
        required=True,
        help="Level of spatial upsampling.",
        type=int,
    )
    parser.add_argument(
        "--temporal_zoom",
        nargs=1,
        metavar="temporal_zoom",
        required=True,
        help="Level of temporal upsampling.",
        type=int,
    )
    parser.add_argument(
        "--time_smoothing_kernel_as_dt_fraction",
        nargs=1,
        metavar="alpha_dt",
        default=[0.5],
        help=(
            "Sigma, expressed as a portion of dt, of the Gaussian kernel used to "
            "smooth SR."
        ),
        type=float,
    )
    parser.add_argument(
        "--use_Yoffe",
        help=(
            "Replace the discretized STF with a Yoffe function (e.g. for comparison "
            "with FL33). Requires peak slip rate threshold (0-1) to determine onset "
            "time and duration of STF."
        ),
        dest="use_Yoffe",
        nargs=1,
        metavar=("PSRthreshold"),
        type=float,
        default=[None],
    )

    args = parser.parse_args()

    yoffePSRthreshold = args.use_Yoffe[0]

    refine(
        args.filename,
        args.mode,
        args.proj,
        yoffePSRthreshold,
        args.spatial_order[0],
        args.spatial_zoom[0],
        args.temporal_zoom[0],
        args.time_smoothing_kernel_as_dt_fraction[0],
    )
