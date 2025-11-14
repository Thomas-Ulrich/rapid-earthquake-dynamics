#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import argparse
import glob
import operator
import re
import socket
import sys
from datetime import datetime
from functools import reduce

import h5py
import numpy as np
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


def parse_timestamp(ts):
    """Try multiple timestamp formats."""
    fmts = [
        "%Y-%m-%d %H:%M:%S.%f",  # new format
        "%a %b %d %H:%M:%S",  # old format
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            pass
    return None


def get_log_info(log_file):
    """
    Extract:
      - runtime_sec: walltime between Welcome and Goodbye
      - kernel_time
      - num_ranks
      - ranks_per_node
    """
    # Regex patterns
    patterns = {
        "kernel_time": re.compile(r"Total time spent in compute kernels.*?(\d+\.?\d*)"),
        "num_ranks": re.compile(r"Using MPI with #ranks:\s*(\d+)"),
        "ranks_per_node": re.compile(r"#ranks/node:\s*(\d+)"),
    }

    # Timestamp patterns (new + old SeisSol)
    welcome_pat = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*Welcome to SeisSol"
        r"|"
        r"([A-Za-z]{3} [A-Za-z]{3} +\d{1,2} \d{2}:\d{2}:\d{2}).*Welcome to SeisSol"
    )
    goodbye_pat = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*SeisSol done"
        r"|"
        r"([A-Za-z]{3} [A-Za-z]{3} +\d{1,2} \d{2}:\d{2}:\d{2}).*SeisSol done"
    )

    # Storage
    extracted = {
        "kernel_time": None,
        "num_ranks": None,
        "ranks_per_node": None,
        "runtime_sec": None,
    }

    welcome_time = None
    goodbye_time = None

    with open(log_file, "r") as f:
        for line in f:
            # ---------- patterns ----------
            for key, pat in patterns.items():
                if extracted[key] is None:
                    m = pat.search(line)
                    if m:
                        extracted[key] = (
                            float(m.group(1))
                            if key == "kernel_time"
                            else int(m.group(1))
                        )

            # ---------- timestamps ----------
            m = welcome_pat.search(line)
            if m:
                ts = m.group(1) or m.group(2)
                welcome_time = parse_timestamp(ts)

            m = goodbye_pat.search(line)
            if m:
                ts = m.group(1) or m.group(2)
                goodbye_time = parse_timestamp(ts)

    # Compute walltime
    if welcome_time and goodbye_time:
        extracted["runtime_sec"] = (goodbye_time - welcome_time).total_seconds()

    return extracted


def convert_to_hms(target_time):
    hours = int(target_time // 3600)
    minutes = int((target_time % 3600) // 60)
    seconds = int(target_time % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_scaled_walltime_and_ranks(
    log_file, simulation_batch_size, max_hours, nodes_config
):
    extracted = get_log_info(args.log_file)
    print("reference run infos:", extracted)
    run_time = extracted["runtime_sec"]
    nodes_ref = extracted["num_ranks"] / extracted["ranks_per_node"]

    # Compute scaled walltime
    min_nodes = nodes_config["min"]
    max_nodes = nodes_config["max"]
    step_nodes = nodes_config["step"]
    max_nodes1 = (step_nodes * simulation_batch_size) // 3
    max_nodes1 = min(max_nodes1, max_nodes)
    candidates = list(range(min_nodes, max_nodes1, step_nodes))

    if simulation_batch_size <= 64:
        # it seems that slurm can get overwhelmed by too many jobs
        nodes_full_batch = step_nodes * simulation_batch_size
        if nodes_full_batch % 2 == 0:
            nodes_half_batch = nodes_full_batch // 2
            if nodes_half_batch <= max_nodes:
                candidates.append(nodes_half_batch)

        if nodes_full_batch <= max_nodes:
            candidates.append(nodes_full_batch)
    print("candidate_nodes: ", candidates)

    chosen_nodes = max_nodes
    walltime = ""
    safety_factor = 1.5

    for nodes in candidates:
        target_time = (
            safety_factor * run_time * simulation_batch_size * nodes_ref / nodes
        )
        hours = int(target_time // 3600)
        walltime = convert_to_hms(target_time)

        chosen_nodes = nodes
        if hours < max_hours:
            break

    return walltime, chosen_nodes


def get_mesh_cells(mesh_fname):
    with h5py.File(mesh_fname, "r") as fid:
        return fid["/connect"].shape[0]


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
    with open("derived_config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    mesh_cells = get_mesh_cells(config_dict["mesh_file"])
    simulation_batch_size = get_simulation_batch_size()

    def get_node_config(
        mesh_cells, simulation_batch_size, target_cell_per_nodes, max_allowed_nodes
    ):
        nodes_per_sim = max(1, int(np.round(mesh_cells / target_cell_per_nodes)))
        max_nodes_raw = min(max_allowed_nodes, nodes_per_sim * simulation_batch_size)
        # Make max_nodes a multiple of nodes_per_sim
        max_nodes = (max_nodes_raw // nodes_per_sim) * nodes_per_sim
        return {"min": nodes_per_sim, "max": max_nodes, "step": nodes_per_sim}

    if args.node_config == "auto":
        if hostname.startswith("uan"):
            # LUMI
            nodes_config = get_node_config(
                mesh_cells,
                simulation_batch_size,
                target_cell_per_nodes=500000,
                max_allowed_nodes=256,
            )
        elif hostname.startswith("login"):
            # supermuc NG
            nodes_config = get_node_config(
                mesh_cells,
                simulation_batch_size,
                target_cell_per_nodes=100000,
                max_allowed_nodes=400,
            )
        else:
            raise ValueError(
                "supercomputer is not among the ones with predefined nodes config"
            )
    else:
        items = [int(x.strip()) for x in args.node_config.split(",")]
        assert len(items) == 3, "wrong number of parameters to unpack in node_config"
        nodes_config = {"min": items[0], "max": items[1], "step": items[2]}
    print("nodes_config:", nodes_config)
    try:
        walltime, chosen_nodes = get_scaled_walltime_and_ranks(
            args.log_file, simulation_batch_size, args.max_hours, nodes_config
        )
        print(f"Walltime: {walltime}")
        print(f"Chosen nodes: {chosen_nodes}")
        print(f"Nodes per sim: {nodes_config['step']}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
