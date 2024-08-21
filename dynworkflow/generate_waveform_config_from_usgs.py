#!/usr/bin/env python3
import json
import os
import argparse
import glob
import jinja2
from dynworkflow.get_usgs_finite_fault_data import (
    find_key_recursive,
    get_value_by_key,
    get_value_from_usgs_data,
)


def generate_waveform_config_file(ignore_source_files=False):
    fn_json = glob.glob("tmp/*.json")[0]

    with open(fn_json) as f:
        jsondata = json.load(f)

    dyfi = get_value_from_usgs_data(jsondata, "dyfi")[0]
    eventtime = dyfi["properties"]["eventtime"]

    finite_fault = get_value_from_usgs_data(jsondata, "finite-fault")[0]
    code_finite_fault = finite_fault["code"]
    hypocenter_x = finite_fault["properties"]["longitude"]
    hypocenter_y = finite_fault["properties"]["latitude"]
    hypocenter_z = finite_fault["properties"]["depth"]

    moment_tensor = get_value_from_usgs_data(jsondata, "moment-tensor")[0]
    duration = moment_tensor["properties"]["sourcetime-duration"]

    with open(f"tmp/projection.txt", "r") as f:
        proj = f.read()

    # Get the directory of the script
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    input_file_dir = f"{script_directory}/input_files"
    templateLoader = jinja2.FileSystemLoader(searchpath=input_file_dir)
    templateEnv = jinja2.Environment(loader=templateLoader)
    if ignore_source_files:
        point_source_files = "{{ point_source_files }}"
    else:
        point_source_files = ",".join(sorted(glob.glob("tmp/PointSou*.h5")))

    template_par = {
        "setup_name": code_finite_fault,
        "source_files": point_source_files,
        "stations": "{{ stations }}",
        "lon": hypocenter_x,
        "lat": hypocenter_y,
        "depth": hypocenter_z,
        "onset": eventtime,
        "t_after_P_onset": 3.0 * float(duration),
        "t_after_SH_onset": 6.0 * float(duration),
        "projection": proj,
    }

    def render_file(template_par, template_fname, out_fname, verbose=True):
        template = templateEnv.get_template(template_fname)
        outputText = template.render(template_par)
        fn_tractions = out_fname
        with open(out_fname, "w") as fid:
            fid.write(outputText)
        if verbose:
            print(f"done creating {out_fname}")

    render_file(template_par, "waveforms_config.tmpl.ini", "waveforms_config.ini")


if __name__ == "__main__":
    generate_waveform_config_file()
