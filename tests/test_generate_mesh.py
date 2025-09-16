# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import os
import shutil
from unittest import mock

from dynworkflow.generate_mesh import generate


@mock.patch("builtins.input", return_value="y")
def test_generate_mesh(mock_input, tmp_path):
    # Arrange path
    tmp_dir = os.path.join(tmp_path, "tmp")
    os.mkdir(tmp_dir)

    for k in [3, 65, 66, 67]:
        ts_file = os.path.abspath(f"tests/data/basic_inversion{k}_fault.ts")
        shutil.copyfile(ts_file, os.path.join(tmp_dir, os.path.basename(ts_file)))

    # Change working directory to tmp_path so that folders and files go there
    cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        # Act
        generate(
            h_domain=20e3, h_fault=20e3, interactive=False, vertex_union_tolerance=500.0
        )
        assert os.path.exists("tmp/mesh2d.stl")
        assert os.path.exists("tmp/mesh.msh")
    finally:
        # Cleanup and restore working dir
        os.chdir(cwd)
