# rapid earthquake dynamics

workflows for automated generation of dynamic rupture scenarios from earthquake kinematic models, enabling rapid source characterization


# Installing requirements

Install and load the easi library with python binding
This can be done, e.g. by installing seissol with:

```bash
spack install -j 8 seissol@master convergence_order=4 dr_quad_rule=dunavant equations=elastic precision=single ^easi +python
# now create a module:
spack module tcl refresh $(spack find -d --format "{name}{/hash:5}" seissol)
module load seissol
```

Then install other requirements:

```bash
python -m pip install -r rapid-earthquake-dynamics/requirements.txt
```

