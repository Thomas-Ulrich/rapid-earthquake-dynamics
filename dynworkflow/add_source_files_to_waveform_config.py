#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import glob
import os
import sys
from pathlib import Path

import step1_args
import yaml
from config_utils import yaml_dump
from seismic_waveform_factory.config.loader import ConfigLoader
from seismic_waveform_factory.config.schema import CONFIG_SCHEMA


def update_file(waveform_type="regional"):
    config_file = f"waveforms_config_{waveform_type}_sources.yaml"
    cfg = ConfigLoader(config_file, CONFIG_SCHEMA)

    # load first default arguments for backwards compatibility
    args = step1_args.get_args()
    input_config = vars(args)

    with open("input_config.yaml", "r") as f:
        input_config |= yaml.safe_load(f)

    regional_synthetics_generator = input_config["regional_synthetics_generator"]

    if waveform_type == "regional" and regional_synthetics_generator == "seissol":
        par_files = glob.glob("parameters_dyn_*.par")

        seissol_outputs = []
        for file in par_files:
            basename = os.path.basename(file)
            # Remove 'parameters_' prefix and '.par' suffix
            code = basename[len("parameters_") : -len(".par")]
            seissol_outputs.append(f"extracted_output/{code}")
        seissol_outputs.sort()
        for syn in cfg["synthetics"]:
            if syn["type"] == "seissol":
                syn["outputs"] = seissol_outputs
    else:
        pattern = f"mps_{waveform_type}/PointSou*.h5"
        point_source_files = sorted(glob.glob(pattern))
        source_files = point_source_files
        for syn in cfg["synthetics"]:
            if syn["type"] != "seissol":
                syn["source_files"] = source_files

    # Save to new YAML file
    # root, ext = os.path.splitext(config_file)
    out_fname = f"waveforms_config_{waveform_type}_sources.yaml"
    print(out_fname)
    yaml_dump(cfg.config, out_fname)


if __name__ == "__main__":
    update_file()
    update_file(waveform_type="teleseismic")
