import numpy as np

from dynworkflow.rank_models import infer_duration


def compute_simulation_end_time(input_config):
    fn_mr = "tmp/moment_rate_from_finite_source_file.txt"
    moment_rate = np.loadtxt(fn_mr)
    kinmod_duration = infer_duration(moment_rate[:, 0], moment_rate[:, 1])
    if input_config["seissol_end_time"] == "auto":
        end_time = kinmod_duration + max(20.0, 0.25 * kinmod_duration)
    else:
        end_time = input_config["seissol_end_time"]
    return float(end_time)
