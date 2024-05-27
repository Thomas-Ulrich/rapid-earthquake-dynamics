#!/usr/bin/env python3
from dynworflow import generate_input_seissol_dr
from dynworkflow.kinematic_models.kinmodmodules import (
    fault_output_generator,
    ugrid_data_projector,
)


if __name__ == "__main__":
    fault_output_generator.generate(
        "output/fl33-fault.xdmf",
        "yaml_files/FL33_34_fault.yaml",
        "output/dyn-kinmod-fault",
        "Gaussian",
        0.5,
    )
    with open(f"tmp/inferred_fault_mesh_size.txt", "r") as f:
        inferred_fault_mesh_size = float(f.read())

    ugrid_data_projector.generate_input_files(
        "output/fl33-fault.xdmf",
        inferred_fault_mesh_size / 2,
        gaussian_kernel=inferred_fault_mesh_size,
        taper=None,
        paraview_readable=None,
    )
    generate_input_seissol_dr.generate()
