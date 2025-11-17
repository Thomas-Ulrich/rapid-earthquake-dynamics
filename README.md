# rapid earthquake dynamics

[![codecov](https://codecov.io/gh/Thomas-Ulrich/rapid-earthquake-dynamics/branch/main/graph/badge.svg)](https://codecov.io/gh/Thomas-Ulrich/rapid-earthquake-dynamics)

Workflows for automated generation of dynamic rupture scenarios from earthquake
fault slip models, enabling rapid source characterization.

## Installing the package

```bash
git clone https://github.com/Thomas-Ulrich/rapid-earthquake-dynamics
cd rapid-earthquake-dynamics
pip install -e .
```

## Installing requirements

### easi library with python bindings

Install and load the easi library with python binding
This can be done, e.g. by installing seissol with:

```bash
spack install -j 8 seissol@master convergence_order=4 dr_quad_rule=dunavant \
    equations=elastic precision=single ^easi +python
# now create a module:
spack module tcl refresh $(spack find -d --format "{name}{/hash:5}" seissol)
# now add the path to the python_wrapper to the python path, e.g. with:
export PYTHONPATH=path_to_spack_installation/linux-sles15-skylake_avx512/easi/1.6.1-gcc-15.1.0-efuzmvl/lib/python3.10/site-packages/easilib/cmake/easi/python_wrapper/:$PYTHONPATH
```

### axitra

Axitra is needed for generating regional waveform synthetics
 (if not using directly SeisSol).

```bash
git clone --branch thomas/remove_logs https://github.com/Thomas-Ulrich/axitra
cd axitra/MOMENT_DISP_F90_OPENMP/src
make all python
```

Finally update `path` under `synthetics` type `axitra` in:
[this file](https://github.com/Thomas-Ulrich/rapid-earthquake-dynamics/blob/main/dynworkflow/input_files/waveforms_config_regional.tmpl.yaml)

## Running the tests locally

```bash
cd rapid-earthquake-dynamics
git lfs install      # Enables Git LFS support for handling large binary files
git lfs pull         # Downloads large files tracked by Git LFS (e.g., .h5, .nc)
docker pull ghcr.io/thomas-ulrich/easi-image:latest
docker run --rm -v "$PWD:/workspace" -w /workspace \
   ghcr.io/thomas-ulrich/easi-image:latest bash -c "pytest --cov=dynworkflow \
   --cov=external --cov=. --cov-report=xml"
```
