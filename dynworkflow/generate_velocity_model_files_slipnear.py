#!/usr/bin/env python3
import pandas as pd

from extract_velocity_model_from_usgs_fsp import (
    write_yaml_material_file,
    write_z_rigidity_to_txt,
)

def generate():

    vel_model = """H P_VEL S_VEL DENS QP QS
0.6 3.3 1.9 2.0 200 100
1.4 4.5 2.6 2.3 350 175
3.0 5.5 3.18 2.5 500 250
25.0 6.5 3.75 2.9 600 300
10000 8.1 4.68 3.3 1000 500"""
    df = pd.read_csv(vel_model, sep=" ")
    df["rho"] = 1000.0 * df["DENS"]
    df["mu"] = 1e6 * df["rho"] * df["S_VEL"] ** 2
    df["lambda"] = 1e6 * df["rho"] * (df["P_VEL"] ** 2 - 2.0 * df["S_VEL"] ** 2)
    df["DEPTH_bot"] = df["H"].cumsum()
    df["DEPTH"] = df["DEPTH_bot"].shift(1)
    df.at[0, "DEPTH"] = -10

    print(df)

    write_yaml_material_file(df)
    write_z_rigidity_to_txt(df)

if __name__ == "__main__":
    generate()
