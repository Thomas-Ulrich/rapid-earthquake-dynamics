# rapid earthquake dynamics

[![codecov](https://codecov.io/gh/Thomas-Ulrich/rapid-earthquake-dynamics/branch/main/graph/badge.svg)](https://codecov.io/gh/Thomas-Ulrich/rapid-earthquake-dynamics)

Workflows for automated generation of dynamic rupture scenarios from earthquake
fault slip models, enabling rapid source characterization.

## Cloning the repository

```bash
git clone https://github.com/Thomas-Ulrich/rapid-earthquake-dynamics
cd rapid-earthquake-dynamics
git lfs install      # Enables Git LFS support for handling large binary files
git lfs pull         # Downloads large files tracked by Git LFS (e.g., .h5, .nc)
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

### other python requirements

Then install other requirements, including seismic-waveform-factory:

```bash
python -m pip install -r rapid-earthquake-dynamics/requirements.txt
python -m pip install git+https://github.com/Thomas-Ulrich/seismic-waveform-factory.git@v0.3.1
```

### axitra

```bash
git clone --branch thomas/build_meson https://github.com/Thomas-Ulrich/axitra
cd axitra/MOMENT_DISP_F90_OPENMP/src
make all python
```

Finally update `path` under `synthetics` type `axitra` in:
[this file](https://github.com/Thomas-Ulrich/rapid-earthquake-dynamics/blob/main/dynworkflow/input_files/waveforms_config_regional.tmpl.yaml)
