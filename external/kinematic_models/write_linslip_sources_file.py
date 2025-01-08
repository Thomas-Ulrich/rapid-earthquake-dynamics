import os
import argparse
from FaultPlane import FaultPlane, MultiFaultPlane


def write_sf(filename, rake):

    prefix, ext = os.path.splitext(filename)
    prefix = os.path.basename(prefix)

    if ext == ".srf":
        mfp = MultiFaultPlane.from_srf(filename)
    elif ext == ".param":
        mfp = MultiFaultPlane.from_usgs_param_file(filename)
    elif ext == ".param2":
        mfp = MultiFaultPlane.from_usgs_param_file_alternative(filename)
    elif ext == ".fsp":
        mfp = MultiFaultPlane.from_usgs_fsp_file(filename)
    elif ext == ".txt":
        mfp = MultiFaultPlane.from_slipnear_param_file(filename)
    else:
        raise NotImplementedError(f" unknown extension: {ext}")
    for p, fp in enumerate(mfp.fault_planes):
        fp.write_linslip_sources_file("sources.dat", rake)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generate linslip source file",
    )
    parser.add_argument("filename", help="filename of the source file")
    parser.add_argument(
        "--rake",
        help="rake written constant instead of read from the kinematic model",
        nargs=1,
        metavar="rake",
        type=float,
    )

    args = parser.parse_args()

    write_sf(args.filename, args.rake)
