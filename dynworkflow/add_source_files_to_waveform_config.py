#!/usr/bin/env python3
import json
import os
import argparse
import glob
import jinja2


def update_file():
    current_directory = os.getcwd()
    templateLoader = jinja2.FileSystemLoader(searchpath=current_directory)
    templateEnv = jinja2.Environment(loader=templateLoader)

    template_par = {}
    point_source_files = ",".join(sorted(glob.glob("tmp/PointSou*.h5")))
    template_par["source_files"] = point_source_files

    def render_file(template_par, template_fname, out_fname, verbose=True):
        template = templateEnv.get_template(template_fname)
        outputText = template.render(template_par)
        fn_tractions = out_fname
        with open(out_fname, "w") as fid:
            fid.write(outputText)
        if verbose:
            print(f"done creating {out_fname}")

    render_file(template_par, "waveforms_config.ini", "waveforms_config_sources.ini")


if __name__ == "__main__":
    update_file()
