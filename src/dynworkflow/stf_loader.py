import numpy as np
import os


def read_usgs_moment_rate(fname: str) -> np.ndarray:
    """Reads and scales USGS moment rate file data."""
    mr_ref = np.loadtxt(fname, skiprows=2)
    # Conversion factor from dyne-cm/sec to Nm/sec (for older usgs files)
    scaling_factor = 1.0 if np.amax(mr_ref[:, 1]) < 1e23 else 1e-7
    mr_ref[:, 1] *= scaling_factor
    return mr_ref


def trim_trailing_zero(mr_ref: np.ndarray) -> np.ndarray:
    """Trims trailing zeros from moment rate array."""
    last_index_non_zero = np.nonzero(mr_ref[:, 1])[0][-1]
    return mr_ref[:last_index_non_zero, :]


def load_reference_stfs(
    derived_config_dict: dict = None,
) -> tuple[list, np.ndarray, str, float, float]:
    """
    Loads reference STF files based on configuration or back-compatibility defaults.

    Returns:
        ref_stfs (list): List of [file_name, label] entries.
        mr_ref (np.ndarray): Primary reference moment rate array.
        ref_name (str): Primary reference name label.
        M0ref (float): Primary reference M0.
        Mwref (float): Primary reference Mw.
    """
    fn = "tmp/reference_STF.txt"
    ref_stfs = []

    if derived_config_dict and "reference_STFs" in derived_config_dict:
        ref_stfs = derived_config_dict["reference_STFs"]
        if not ref_stfs:
            raise ValueError(
                "reference_STFs not found in derived_config.yaml. "
                "If using an old setup, change the reference_STF entry to a "
                "reference_STFs: [[file_name, label]] format."
            )
        first_ref = ref_stfs[0]
        refMRFfile, ref_name = first_ref
    elif os.path.exists(fn):
        with open(fn, "r") as fid:
            refMRFfile = fid.read().strip()
        ref_name = "finite-source model"
    else:
        ref_name = "finite-source model"
        if os.path.exists("tmp/moment_rate_from_finite_source_file_usgs.txt"):
            refMRFfile = "tmp/moment_rate.mr"
        else:
            refMRFfile = "tmp/moment_rate_from_finite_source_file.txt"

    if refMRFfile == "tmp/moment_rate_from_finite_source_file.txt":
        mr_ref = np.loadtxt("tmp/moment_rate_from_finite_source_file.txt")
    else:
        mr_ref = read_usgs_moment_rate(refMRFfile)

    mr_ref = trim_trailing_zero(mr_ref).astype(np.float64)

    # Compute M0, Mw for main ref
    M0ref = np.trapezoid(mr_ref[:, 1], x=mr_ref[:, 0])
    Mwref = 2.0 * np.log10(M0ref) / 3.0 - 6.07

    return ref_stfs, mr_ref, ref_name, M0ref, Mwref
