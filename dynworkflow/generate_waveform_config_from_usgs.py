#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import glob
import json
import os

import jinja2
import yaml

try:
    # Try relative import if called from the full package
    from dynworkflow.get_usgs_finite_fault_data import get_value_from_usgs_data
except ImportError:
    # Fallback to local import if run directly
    from get_usgs_finite_fault_data import get_value_from_usgs_data


def generate_waveform_config_file(
    regional_stations="auto", teleseismic_stations="auto", ignore_source_files=False
):
    fn_json = glob.glob("tmp/*.json")[0]

    with open(fn_json) as f:
        jsondata = json.load(f)

    code = get_value_from_usgs_data(jsondata, "code")
    origin = get_value_from_usgs_data(jsondata, "origin")
    preferred = max(range(len(origin)), key=lambda k: origin[k]["preferredWeight"])
    hypocenter_x = float(origin[preferred]["properties"]["longitude"])
    hypocenter_y = float(origin[preferred]["properties"]["latitude"])
    hypocenter_z = float(origin[preferred]["properties"]["depth"])
    eventtime = origin[preferred]["properties"]["eventtime"]

    moment_tensor = get_value_from_usgs_data(jsondata, "moment-tensor")[0]
    duration = float(moment_tensor["properties"]["sourcetime-duration"])

    with open("derived_config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    proj = config_dict["projection"]

    # Get the directory of the script
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    input_file_dir = f"{script_directory}/input_files"
    templateLoader = jinja2.FileSystemLoader(searchpath=input_file_dir)
    templateEnv = jinja2.Environment(loader=templateLoader)
    comparison_duration = max(1.3 * duration, duration + 20)
    template_par = {
        "setup_name": code,
        "lon": hypocenter_x,
        "lat": hypocenter_y,
        "depth": hypocenter_z,
        "onset": eventtime,
        "t_after_P_onset": comparison_duration,
        "t_after_SH_onset": comparison_duration,
        "projection": proj,
    }

    if not ignore_source_files:
        point_source_files = ", ".join(sorted(glob.glob("tmp/PointSou*.h5")))
        template_par["source_files"] = point_source_files
    else:
        template_par["source_files"] = ""
        template_par["seissol_outputs"] = ""
        """
        template_par[
            "source_files"
        ] = "{{ source_files | default('{{ source_files }}', true) }}"
        template_par[
            "seissol_outputs"
        ] = "{{ seissol_outputs | default('{{ seissol_outputs }}', true) }}"
        """

    def render_file(template_par, template_fname, out_fname, verbose=True):
        template = templateEnv.get_template(template_fname)
        outputText = template.render(template_par)
        with open(out_fname, "w") as fid:
            fid.write(outputText)
        if verbose:
            print(f"done creating {out_fname}")

    for name, user_stations in [
        ("regional", regional_stations),
        ("teleseismic", teleseismic_stations),
    ]:
        template_par["stations"] = "" if user_stations == "auto" else user_stations
        render_file(
            template_par,
            f"waveforms_config_{name}.tmpl.yaml",
            f"waveforms_config_{name}.yaml",
        )


if __name__ == "__main__":
    with open("input_config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    regional_seismic_stations = config_dict["regional_seismic_stations"]
    teleseismic_stations = config_dict["teleseismic_stations"]
    generate_waveform_config_file(regional_seismic_stations, teleseismic_stations, True)
