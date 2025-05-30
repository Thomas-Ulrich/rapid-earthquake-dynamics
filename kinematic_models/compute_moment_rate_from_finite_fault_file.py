#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import easi
import os
import argparse
import os.path
import numpy as np
from stf import gaussianSTF
from multi_fault_plane import MultiFaultPlane


def compute(filename, yaml_filename, projection, dt=0.5, tmax=None):
    prefix, ext = os.path.splitext(filename)
    prefix = os.path.basename(prefix)
    mfp = MultiFaultPlane.from_file(filename)
    if tmax:
        mfp.temporal_crop(tmax)

    duration = 0
    for p, fp in enumerate(mfp.fault_planes):
        duration = max(duration, fp.t0.max() + fp.rise_time.max())
    fp = mfp.fault_planes[0]
    has_STF = fp.ndt > 0
    if has_STF:
        print(f"STF described by time series in {filename}")
        time = fp.myt
    else:
        print(f"a Gaussian STF will be used for {filename}")
        time = np.arange(0, duration, dt)
    moment_rate = np.zeros_like(time)

    for p, fp in enumerate(mfp.fault_planes):
        fp.compute_xy_from_latlon(projection)
        centers = np.column_stack(
            (fp.x.flatten(), fp.y.flatten(), -fp.depth.flatten() * 1e3)
        )
        tags = np.zeros_like(centers[:, 0]) + 1
        out = easi.evaluate_model(
            centers,
            tags,
            ["mu"],
            yaml_filename,
        )

        mu = out["mu"].reshape(fp.x.shape)
        for k, tk in enumerate(time):
            if not has_STF:
                STF = gaussianSTF(tk - fp.t0[:, :], fp.rise_time[:, :], dt)
            for j in range(fp.ny):
                for i in range(fp.nx):
                    STFij = fp.aSR[j, i, k] if has_STF else STF[j, i]
                    moment_rate[k] += (
                        mu[j, i] * fp.dx * fp.dy * 1e6 * STFij * fp.slip1[j, i] * 0.01
                    )
    M0 = np.trapz(moment_rate[:], x=time[:])
    Mw = 2.0 * np.log10(M0) / 3.0 - 6.07
    print(f"inferred Mw {Mw} and duration {duration}:")

    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    fname = "tmp/moment_rate_from_finite_source_file.txt"
    with open(fname, "w") as f:
        np.savetxt(f, np.column_stack((time, moment_rate)), fmt="%g")
    print(f"done writing {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "compute moment rate release from kinematic model. Assumptions: Gaussian "
            "source time function, no bimaterial conditions"
        )
    )
    parser.add_argument(
        "--dt",
        nargs=1,
        metavar="dt",
        default=[0.25],
        help="sampling time of the output file",
        type=float,
    )

    parser.add_argument("filename", help="filename of the srf file")
    parser.add_argument("yaml_filename", help="fault easi/yaml filename")

    parser.add_argument(
        "--proj",
        metavar="proj",
        nargs=1,
        help="proj4 string describing the projection",
        required=True,
    )
    args = parser.parse_args()
    compute(args.filename, args.yaml_filename, args.proj[0], args.dt[0])
