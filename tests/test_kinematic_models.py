# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import os
import shutil
from pathlib import Path
from unittest import mock

from kinematic_models import (
    compute_moment_rate_function,
    generate_fault_output_from_fl33_input_files,
)


@mock.patch("builtins.input", return_value="y")
def test_compute_moment_rate_function(mock_input, tmp_path: Path):
    # Arrange paths
    tmp_dir = tmp_path / "tmp"
    yaml_dir = tmp_path / "yaml_files"
    tmp_dir.mkdir(exist_ok=True)
    yaml_dir.mkdir(exist_ok=True)

    # Source files
    source_dir = Path("tests/data/us6000jlqa_multifaults")

    # Copy files to temp dirs
    shutil.copy(
        source_dir / "tmp/basic_inversion.param", tmp_dir / "basic_inversion.param"
    )
    shutil.copy(source_dir / "yaml_files/material.yaml", yaml_dir / "material.yaml")

    # Change working directory to tmp_path
    cwd = Path.cwd()
    os.chdir(tmp_path)

    try:
        filename = "tmp/basic_inversion.param"
        yaml_filename = "yaml_files/material.yaml"
        projection = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=37.20 +lat_0=38.01"

        compute_moment_rate_function.compute(
            filename, yaml_filename, projection, dt=0.5, tmax=None
        )

        # Assert output file exists
        assert (tmp_path / "tmp/moment_rate_from_finite_source_file.txt").exists()
    finally:
        # Restore working directory
        os.chdir(cwd)


@mock.patch("builtins.input", return_value="y")
def test_generate_fault_output_from_fl33_input_files(mock_input, tmp_path: Path):
    # Arrange paths
    tmp_dir = tmp_path / "tmp"
    yaml_dir = tmp_path / "yaml_files"
    ASAGI_dir = tmp_path / "ASAGI_files"
    tmp_dir.mkdir(exist_ok=True)
    yaml_dir.mkdir(exist_ok=True)
    ASAGI_dir.mkdir(exist_ok=True)

    # Source files
    source_dir = Path("tests/data/us6000jlqa_multifaults")

    # Copy files to temp dirs
    shutil.copy(source_dir / "tmp/mesh_bc_faults.xdmf", tmp_dir / "mesh_bc_faults.xdmf")
    shutil.copy(source_dir / "tmp/mesh_bc_faults.h5", tmp_dir / "mesh_bc_faults.h5")
    shutil.copytree(source_dir / "yaml_files", yaml_dir, dirs_exist_ok=True)
    shutil.copytree(source_dir / "ASAGI_files", ASAGI_dir, dirs_exist_ok=True)

    # Change working directory to tmp_path
    cwd = Path.cwd()
    os.chdir(tmp_path)

    try:
        generate_fault_output_from_fl33_input_files.generate(
            fault_filename="tmp/mesh_bc_faults.xdmf",
            yaml_filename="yaml_files/FL33_34_fault.yaml",
            output_file="fault_from_fl33_input",
            stf="Gaussian",
            dt_output=20.0,
        )

        # Assert output file exists
        assert (tmp_path / "fault_from_fl33_input.h5").exists()
    finally:
        os.chdir(cwd)
