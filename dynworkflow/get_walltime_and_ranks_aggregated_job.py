#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import argparse
import glob
import operator
import re
import socket
import sys
from functools import reduce

import yaml


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
    print("parameters:", input_config)
    return input_config


def get_simulation_batch_size():
    with open("input_config.yaml", "r") as f:
        input_config = yaml.safe_load(f)
    parameters_structured = parse_parameter_string(input_config["parameters"])
    n_list = [len(parameters_structured[x]) for x in parameters_structured]
    product = reduce(operator.mul, n_list, 1)
    return product


def extract_patterns_from_files(files, patterns):
    """
    Search through a list of files for multiple regex patterns.

    Args:
        files (list of str): List of file paths to search.
        patterns (dict): Dictionary of {key: regex_pattern} to extract.

    Returns:
        dict: {key: extracted_value} for each pattern found.

    Raises:
        ValueError: If any pattern is not found in any file.
    """
    results = {key: None for key in patterns}

    for file in files:
        with open(file, "r") as f:
            for line in f:
                for key, pattern in patterns.items():
                    if results[key] is None:
                        match = pattern.search(line)
                        if match:
                            results[key] = match.group(1)
                # Stop early if all values found
                if all(value is not None for value in results.values()):
                    break
        if all(value is not None for value in results.values()):
            break

    # Convert numeric values where appropriate and check missing
    for key, value in results.items():
        if value is None:
            raise ValueError(f"Could not extract {key} from files {files}")
        if key in ["kernel_time"]:
            results[key] = float(value)
        else:
            results[key] = int(value)

    return results


def convert_to_hms(target_time):
    hours = int(target_time // 3600)
    minutes = int((target_time % 3600) // 60)
    seconds = int(target_time % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_scaled_walltime_and_ranks(
    log_file, simulation_batch_size, max_hours, nodes_config
):
    files = glob.glob(args.log_file)
    patterns = {
        "kernel_time": re.compile(r"Total time spent in compute kernels.*?(\d+\.?\d*)"),
        "num_ranks": re.compile(r"Using MPI with #ranks:\s*(\d+)"),
        "ranks_per_node": re.compile(r"#ranks/node:\s*(\d+)"),
    }

    extracted = extract_patterns_from_files(files, patterns)
    print("reference run infos:", extracted)
    # Output: {'kernel_time': 123.45, 'num_ranks': 16, 'ranks_per_node': 8}
    kernel_time = extracted["kernel_time"]
    nodes_ref = extracted["num_ranks"] / extracted["ranks_per_node"]

    # Compute scaled walltime
    min_nodes = nodes_config["min"]
    max_nodes = nodes_config["max"]
    step_nodes = nodes_config["step"]
    candidates = list(range(min_nodes, max_nodes + 1, step_nodes))
    chosen_nodes = max_nodes
    walltime = ""
    safety_factor = 1.5

    for nodes in candidates:
        target_time = (
            safety_factor * kernel_time * simulation_batch_size * nodes_ref / nodes
        )
        hours = int(target_time // 3600)
        walltime = convert_to_hms(target_time)

        chosen_nodes = nodes
        if hours <= max_hours:
            break

    return walltime, chosen_nodes


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate wall time and number of nodes given fl33 log file"
    )
    parser.add_argument("log_file", type=str, help="log file from the fl33 run")
    parser.add_argument(
        "--max_hours", type=float, help="max wall time in hours", default=3.0
    )
    parser.add_argument(
        "--node_config",
        help=(
            "coma separated min,max,step number of nodes to be used, or auto,"
            "in which case these numbers are hardcoded depended on the host name"
            " (Lumi or supermucNG 1)"
        ),
        default="auto",
    )
    args = parser.parse_args()
    hostname = socket.gethostname()
    if args.node_config == "auto":
        if hostname.startswith("uan"):
            # LUMI
            nodes_config = {"min": 2, "max": 256, "step": 2}
        elif hostname.startswith("login"):
            # supermuc NG
            nodes_config = {"min": 32, "max": 400, "step": 16}
        else:
            raise ValueError(
                "supercomputer is not among the ones with predefined nodes config"
            )
    else:
        items = [int(x.strip()) for x in args.node_config.split(",")]
        assert len(items) == 3, "wrong number of parameters to unpack in node_config"
        nodes_config = {"min": items[0], "max": items[1], "step": items[2]}
    print("nodes_config:", nodes_config)
    simulation_batch_size = get_simulation_batch_size()
    try:
        walltime, chosen_nodes = get_scaled_walltime_and_ranks(
            args.log_file, simulation_batch_size, args.max_hours, nodes_config
        )
        print(f"Walltime: {walltime}")
        print(f"Chosen nodes: {chosen_nodes}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
