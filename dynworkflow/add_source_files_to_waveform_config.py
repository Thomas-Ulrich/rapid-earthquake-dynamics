#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import os
import glob
import jinja2
import yaml
from dynworkflow import step1_args


def update_file():
    current_directory = os.getcwd()
    templateLoader = jinja2.FileSystemLoader(searchpath=current_directory)
    templateEnv = jinja2.Environment(loader=templateLoader)
    template_par = {}

    # load first default arguments for backwards compatibility
    args = step1_args.get_args()
    input_config = vars(args)

    with open("input_config.yaml", "r") as f:
        input_config |= yaml.safe_load(f)
    regional_synthetics_generator = input_config["regional_synthetics_generator"]

    if regional_synthetics_generator == "seissol":
        par_files = glob.glob("parameters_dyn_*.par")

        seissol_outputs = []
        for file in par_files:
            basename = os.path.basename(file)
            # Remove 'parameters_' prefix and '.par' suffix
            code = basename[len("parameters_") : -len(".par")]
            seissol_outputs.append(f"output/{code}")
        seissol_outputs.sort()
        template_par["source_files"] = " "
        template_par["seissol_outputs"] = ",".join(seissol_outputs)
    else:
        point_source_files = ",".join(sorted(glob.glob("tmp/PointSou*.h5")))
        template_par["source_files"] = point_source_files
        template_par["seissol_outputs"] = " "

    def render_file(template_par, template_fname, out_fname, verbose=True):
        template = templateEnv.get_template(template_fname)
        outputText = template.render(template_par)
        with open(out_fname, "w") as fid:
            fid.write(outputText)
        if verbose:
            print(f"done creating {out_fname}")

    render_file(template_par, "waveforms_config.ini", "waveforms_config_sources.ini")


if __name__ == "__main__":
    update_file()
