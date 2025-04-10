#!/usr/bin/env python3
import json
import os
import glob
import jinja2
from dynworkflow.get_usgs_finite_fault_data import (
    get_value_from_usgs_data,
)


def generate_waveform_config_file(ignore_source_files=False):
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
    duration = moment_tensor["properties"]["sourcetime-duration"]

    with open("tmp/projection.txt", "r") as f:
        proj = f.read()

    # Get the directory of the script
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    input_file_dir = f"{script_directory}/input_files"
    templateLoader = jinja2.FileSystemLoader(searchpath=input_file_dir)
    templateEnv = jinja2.Environment(loader=templateLoader)

    template_par = {
        "setup_name": code,
        "stations": "{{ stations }}",
        "lon": hypocenter_x,
        "lat": hypocenter_y,
        "depth": hypocenter_z,
        "onset": eventtime,
        "t_after_P_onset": 3.0 * float(duration),
        "t_after_SH_onset": 6.0 * float(duration),
        "projection": proj,
    }

    if not ignore_source_files:
        point_source_files = ",".join(sorted(glob.glob("tmp/PointSou*.h5")))
        template_par["source_files"] = point_source_files
    else:
        template_par["source_files"] = (
            "{{ source_files | default('{{ source_files }}', true) }}"
        )

    def render_file(template_par, template_fname, out_fname, verbose=True):
        template = templateEnv.get_template(template_fname)
        outputText = template.render(template_par)
        with open(out_fname, "w") as fid:
            fid.write(outputText)
        if verbose:
            print(f"done creating {out_fname}")

    render_file(template_par, "waveforms_config.tmpl.ini", "waveforms_config.ini")


if __name__ == "__main__":
    generate_waveform_config_file()
