# rapid earthquake dynamics

workflows for automated generation of dynamic rupture scenarios from earthquake kinematic models, enabling rapid source characterization


# Installing requirements

## easi library with python bindings

Install and load the easi library with python binding
This can be done, e.g. by installing seissol with:

```bash
spack install -j 8 seissol@master convergence_order=4 dr_quad_rule=dunavant equations=elastic precision=single ^easi +python
# now create a module:
spack module tcl refresh $(spack find -d --format "{name}{/hash:5}" seissol)
module load seissol
```

## other python requirements

Then install other requirements:

```bash
python -m pip install -r rapid-earthquake-dynamics/requirements.txt
```

## axitra

```
git clone https://github.com/coutanto/axitra
cd axitra/MOMENT_DISP_F90_OPENMP
make all python
```

Finally update axitra_path in
https://github.com/Thomas-Ulrich/rapid-earthquake-dynamics/blob/main/dynworkflow/input_files/waveforms_config.tmpl.ini
