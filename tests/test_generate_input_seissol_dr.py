# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import os
import shutil
import pytest
import yaml
import sys
from unittest.mock import patch
import glob

# Append kinematic_models folder to path
# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "../dynworkflow/kinematic_models"
absolute_path = os.path.join(current_script_dir, relative_path)
if absolute_path not in sys.path:
    sys.path.append(absolute_path)

import project_fault_tractions_onto_asagi_grid

from dynworkflow import generate_input_seissol_dr


@pytest.fixture
def step2_test_env(tmp_path):
    # Absolute source data path
    source_dir = os.path.abspath("tests/data/step2")

    # Copy all contents of source_dir into tmp_path directly
    for item in os.listdir(source_dir):
        s = os.path.join(source_dir, item)
        d = tmp_path / item
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

    # Change to temp working directory
    old_cwd = os.getcwd()
    os.chdir(tmp_path)

    yield tmp_path

    # Restore working directory
    os.chdir(old_cwd)


def test_step2_workflow(step2_test_env):
    # Load config
    with open("derived_config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    fault_mesh_size = config_dict["fault_mesh_size"]

    # Find correct fl33_file
    fl33_file = "output/fl33-fault.xdmf"
    if not os.path.exists(fl33_file):
        fl33_file = "extracted_output/fl33_extracted-fault.xdmf"

    # Run the functions
    project_fault_tractions_onto_asagi_grid.generate_input_files(
        fl33_file,
        fault_mesh_size / 2,
        use_median_of_n_time_steps=7,
        gaussian_kernel=fault_mesh_size,
        taper=None,
        paraview_readable=None,
    )

    assert os.path.exists("yaml_files/Ts0Td0.yaml")
    for k in range(1, 5):
        assert os.path.exists(f"ASAGI_files/basic_inversion{k}_3_cubic.nc")

    # Now patch compute_critical_nucleation inside generate_input_seissol_dr
    with patch(
        "dynworkflow.generate_input_seissol_dr.compute_critical_nucleation"
    ) as mock_compute:
        # Define the fake output
        mock_compute.side_effect = lambda fl33_file, *_args, **_kwargs: [
            1000 for _ in range(len(fl33_file))
        ]

        # Now call generate() with the mock active
        generate_input_seissol_dr.generate()

    # Check that there are 12 parameters files
    param_files = sorted(glob.glob("parameters_dyn_*.par"))
    assert (
        len(param_files) == 12
    ), f"Expected 12 parameter files, found {len(param_files)}"

    # Optionally, check that they are named correctly
    for i, param_file in enumerate(param_files):
        expected_prefix = f"parameters_dyn_{i:04d}_"
        assert os.path.basename(param_file).startswith(
            expected_prefix
        ), f"Unexpected filename: {param_file}"
