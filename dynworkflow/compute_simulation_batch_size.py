#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import yaml
import re
from functools import reduce
import operator


def parse_parameter_string(param_str):
    param_str = param_str.replace("_", "")
    input_config = {}
    for match in re.finditer(r"(\w+)=([^\s]+)", param_str):
        key, val = match.group(1), match.group(2)
        if key == "cohesion":
            input_config["cohesion"] = [
                list(map(float, pair.split(","))) for pair in val.split(";")
            ]
        else:
            input_config[key] = [float(v) for v in val.split(",") if v.strip()]
    print(input_config)
    return input_config


with open("input_config.yaml", "r") as f:
    input_config = yaml.safe_load(f)
parameters_structured = parse_parameter_string(input_config["parameters"])
n_list = [len(parameters_structured[x]) for x in parameters_structured]
product = reduce(operator.mul, n_list, 1)
print(f"simulation_batch_size={product}")
