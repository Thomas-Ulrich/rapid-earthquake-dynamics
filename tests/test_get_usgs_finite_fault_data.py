# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import os
import shutil
from unittest import mock

from dynworkflow.get_usgs_finite_fault_data import get_data


@mock.patch("dynworkflow.get_usgs_finite_fault_data.wget_overwrite")
@mock.patch("builtins.input", return_value="y")
def test_get_data_with_mocked_usgs(mock_input, mock_wget, tmp_path):
    # Arrange paths
    test_json_path = os.path.abspath("tests/data/us7000pn9s.json")
    temp_json = tmp_path / "us7000pn9s.json"
    shutil.copyfile(test_json_path, temp_json)

    # Mock wget_overwrite to just "download" the JSON into working dir
    def fake_wget(url, output_file):
        shutil.copyfile(test_json_path, output_file)

    mock_wget.side_effect = fake_wget

    # Change working directory to tmp_path so that folders and files go there
    cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        # Act
        result = get_data(
            usgs_id_or_dtgeo_npy="us7000pn9s",
            min_magnitude=6.0,
            suffix="_test",
            use_usgs_finite_fault=False,
            download_usgs_fsp=False,
        )

        # Assert
        assert "folder_name" in result
        assert "projection" in result
        assert (
            result["folder_name"]
            == "2025-03-28_Mw7.7_Burma_Myanmar_Earthq_us7000pn9s_test"
        )
        assert (
            result["projection"]
            == "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=95.92 +lat_0=22.00"
        )
        assert os.path.exists(result["folder_name"])

    finally:
        # Cleanup and restore working dir
        os.chdir(cwd)
