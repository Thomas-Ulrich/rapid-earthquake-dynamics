#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import io
import os

import pandas as pd


def write_yaml_material_file(df):
    to_write = """
!LayeredModel
map: !AffineMap
  matrix:
    z: [0.0, 0.0, 1.0]
  translation:
    z: 0
interpolation: upper
parameters: [rho, mu, lambda, Qp, Qs]
nodes:\n"""
    for index, row in df.iterrows():
        to_write += (
            f"   {-1e3 * row['DEPTH']}:"
            f" [{row['rho']},{row['mu']:.10e},{row['lambda']:.10e}, {row['QP']},"
            f" {row['QS']}]"
        )
        to_write += f" #[{row['P_VEL']}, {row['S_VEL']}]\n"

    fname = "yaml_files/material.yaml"
    with open(fname, "w") as fid:
        fid.write(to_write)
    print(f"done writing {fname}")


def write_z_rigidity_to_txt(df):
    """to be used in the teleseismic routines"""
    to_write = ""
    G_prev = False
    eps = 1e-5
    for index, row in df.iterrows():
        if G_prev:
            to_write += f"{-1e3 * row['DEPTH'] - eps} {G_prev:.10e}\n"
        to_write += f"{-1e3 * row['DEPTH']} {row['mu']:.10e}\n"
        G_prev = row["mu"]
    to_write += f"-1e10 {G_prev:.10e}\n"
    fname = "tmp/depth_vs_rigidity.txt"
    with open(fname, "w") as fid:
        fid.write(to_write)
    print(f"done writing {fname}")


def write_axitra_velocity_file(df):
    """to be used in axitra"""
    to_write = "# layer_width Vp Vs rho Qp Qs\n"
    for index, row in df.iterrows():
        h = row["H"] if row["H"] < 1e4 else 0
        to_write += (
            f"{1000 * h:.18e} {1000 * row['P_VEL']:.18e} "
            f"{1000 * row['S_VEL']:.18e} {1000 * row['DENS']:.18e} "
            f"{row['QP']:.18e} {row['QS']:.18e}\n"
        )
    fname = "tmp/axitra_velocity_model.txt"
    print(to_write)
    with open(fname, "w") as fid:
        fid.write(to_write)
    print(f"done writing {fname}")


vel_model_slipnear = """H P_VEL S_VEL DENS QP QS
0.6 3.3 1.9 2.0 200 100
1.4 4.5 2.6 2.3 350 175
3.0 5.5 3.18 2.5 500 250
25.0 6.5 3.75 2.9 600 300
10000 8.1 4.68 3.3 1000 500"""


def generate_arbitrary_velocity_files(vel_model=vel_model_slipnear):
    if os.path.isfile(vel_model):
        with open(vel_model, "r") as file:
            vel_model_content = file.read()
    else:
        vel_model_content = vel_model

    df = pd.read_csv(io.StringIO(vel_model_content), sep=" ")
    df["rho"] = 1000.0 * df["DENS"]
    df["mu"] = 1e6 * df["rho"] * df["S_VEL"] ** 2
    df["lambda"] = 1e6 * df["rho"] * (df["P_VEL"] ** 2 - 2.0 * df["S_VEL"] ** 2)
    df["DEPTH_bot"] = df["H"].cumsum()
    df["DEPTH"] = df["DEPTH_bot"].shift(1)
    df.at[0, "DEPTH"] = -10

    print(df)

    write_yaml_material_file(df)
    write_z_rigidity_to_txt(df)
    write_axitra_velocity_file(df)


def read_velocity_model_from_fsp_file(fname):
    from io import StringIO

    import pandas as pd

    with open(fname, "r") as fid:
        lines = fid.readlines()

    if "FINITE-SOURCE RUPTURE MODEL" not in lines[0]:
        raise ValueError("Not a valid USGS fsp file.")

    def read_param(line, name, dtype=int):
        if name not in line:
            raise ValueError(f"{name} not found in line: {line}")
        else:
            return dtype(line.split(f"{name} =")[1].split()[0])

    def get_to_first_line_starting_with(lines, pattern):
        for i, line in enumerate(lines):
            if line.startswith(pattern):
                return lines[i:]
        raise ValueError(f"{pattern} not found")

    lines = get_to_first_line_starting_with(lines, "% VELOCITY-DENSITY")
    nlayers = read_param(lines[1], "layers")
    text_file = StringIO("\n".join(lines[3 : 5 + nlayers]))

    df = pd.read_csv(text_file, sep=r"\s+").drop([0])
    df = df.apply(pd.to_numeric, errors="coerce")
    rows_to_remove = df[df["DEPTH"] == 0]
    print("removing row\n", rows_to_remove)
    df = df[df["DEPTH"] > 0].reset_index(drop=True)
    if "S-VEL" in df:
        # old usgs format
        df.rename(columns={"S-VEL": "S_VEL"}, inplace=True)
        df.rename(columns={"P-VEL": "P_VEL"}, inplace=True)

    df["rho"] = 1000.0 * df["DENS"]
    df["mu"] = 1e6 * df["rho"] * df["S_VEL"] ** 2
    df["lambda"] = 1e6 * df["rho"] * (df["P_VEL"] ** 2 - 2.0 * df["S_VEL"] ** 2)
    df["H"] = 10000.0

    df.at[0, "DEPTH"] = 0
    # in this file, DEPTH is the top layer depth
    for index, row in df.iterrows():
        if index < len(df) - 1:
            df.loc[index, "H"] = df["DEPTH"].iloc[index + 1] - df["DEPTH"].iloc[index]

    df.at[0, "DEPTH"] = -10.0
    print(df)
    return df


def generate_usgs_velocity_files():
    df = read_velocity_model_from_fsp_file("tmp/complete_inversion.fsp")
    write_yaml_material_file(df)
    write_z_rigidity_to_txt(df)
    write_axitra_velocity_file(df)


if __name__ == "__main__":
    generate_usgs_velocity_files()
