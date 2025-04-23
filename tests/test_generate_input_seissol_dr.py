import os
import shutil
import tempfile
import pytest
import yaml
import sys

# Append kinematic_models folder to path
# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "../dynworkflow/kinematic_models"
absolute_path = os.path.join(current_script_dir, relative_path)
if absolute_path not in sys.path:
    sys.path.append(absolute_path)

import project_fault_tractions_onto_asagi_grid

# from dynworkflow import generate_input_seissol_dr


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
        gaussian_kernel=fault_mesh_size,
        taper=None,
        paraview_readable=None,
    )

    assert os.path.exists("yaml_files/Ts0Td0.yaml")
    for k in range(1, 5):
        assert os.path.exists(f"ASAGI_files/basic_inversion{k}_3_cubic.nc")

    # not activated yet, because of easi module
    # generate_input_seissol_dr.generate()
