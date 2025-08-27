import argparse
import sys


def get_parser():
    parser = argparse.ArgumentParser(
        description="""
        Automatically set up an ensemble of dynamic rupture models from a kinematic
        finite fault model.

        You can either:
        1. Provide all parameters via command-line arguments, or
        2. Use the --config option to load parameters from a YAML config file.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML config file containing all input parameters.",
    )

    parser.add_argument(
        "--custom_setup_files",
        type=lambda s: s.split(";"),
        default=[],
        help="""
           Semicolon-separated list of files (e.g., material.yaml;data.nc) to
           overwrite default files in the earthquake setup folder. yaml files
           will be copied to yaml_files folder. nc files to ASAGI_files folder
        """,
    )

    parser.add_argument(
        "--event_id",
        type=str,
        help="""
        Earthquake event identifier.
        - If using USGS, this is the event ID (e.g., 'us6000d3zh').
        - If using a custom model, this can be a local event dictionary name.
        """,
    )

    parser.add_argument(
        "--fault_receiver_file",
        type=str,
        default=None,
        help="Path to a fault receiver file",
    )

    parser.add_argument(
        "--fault_reference",
        type=str,
        metavar="X,Y,Z,METHOD",
        default="-0.1,0,1.0,1",
        help="""
        Comma-separated reference vector and method:
         X,Y,Z,METHOD (method: 0=point, 1=direction)
        """,
    )

    parser.add_argument(
        "--fault_mesh_size",
        type=str,
        default="auto",
        help="""
        auto: inferred from fault dimensions
        else provide a value
        """,
    )

    parser.add_argument(
        "--finite_fault_model",
        type=str,
        default="usgs",
        help="Path to an alternative finite fault model file.",
    )
    parser.add_argument(
        "--gmsh_vertex_union_tolerance",
        help="minimum distance below which vertices are merged",
        type=float,
        default=500.0,
    )

    parser.add_argument(
        "--gof_components",
        type=str,
        default="slip_distribution,regional_wf,moment_rate_function",
        help=(
            "Comma-separated list of goodness-of-fit components to use for model "
            "validation. Valid options: slip_distribution, teleseismic_body_wf, "
            "teleseismic_surface_wf, regional_wf, moment_rate_function, "
            "fault_offsets, seismic_moment. "
            "An optional weight can be assigned to each component (default is 1.0). "
            "Example: 'slip_distribution 2.0, teleseismic_body_wf' will assign double "
            "the weight to slip_distribution compared to teleseismic_body_wf."
        ),
    )

    parser.add_argument(
        "--hypocenter",
        type=str,
        default="finite_fault",
        help="""
        Specify the hypocenter location. Options:
        - finite_fault: use the first rupturing point in the finite-fault model.
        - usgs: use the most recent origin coordinates from the USGS.
        - lon,lat,depth_km: manually provide coordinates (e.g., '86.08,27.67,15').
        """,
    )

    parser.add_argument(
        "--mesh",
        type=str,
        default="auto",
        help="Path to an alternative mesh file",
    )

    parser.add_argument(
        "--mode",
        choices=["grid_search", "latin_hypercube", "picked_models"],
        default="grid_search",
        help="Sampling strategy for DR input generation.",
    )

    parser.add_argument(
        "--parameters",
        type=str,
        default=(
            "B=0.9,1.0,1.1,1.2 C=0.1,0.2,0.3,0.4,0.5 "
            "R=0.55,0.6,0.65,0.7,0.8,0.9 cohesion=0.25,1,6"
        ),
        help=(
            "Parameter definitions in 'key=val1,val2 ...' format. "
            "Separate key-value pairs with spaces. For cohesion, use "
            "semicolon-separated tuples, with 3 values K0,K1 (MPa) and d_coh (km)"
            "K(z)  = K0 + K1 max(d-d_coh/d_coh))"
            "e.g. B=0.2,0.3 C=0.1,0.2,0.3 R=0.7,0.8 cohesion=0.25,0,6;0.3,1,6"
        ),
    )

    parser.add_argument(
        "--projection",
        type=str,
        default="auto",
        help="""
        Map projection specification.
        - 'auto': transverse Mercator centered on the USGS first estimated hypocenter.
        - OR: custom projection string in Proj4 format
        (e.g., '+proj=utm +zone=33 +datum=WGS84').
        """,
    )

    parser.add_argument(
        "--reference_moment_rate_function",
        type=str,
        default="auto",
        help="""
        Reference moment rate function (used for model ranking).
        - 'auto': download STF from USGS if available, or infer from finite fault model.
        - OR: path to a 2-column STF file in USGS format (2 lines header).
        """,
    )

    parser.add_argument(
        "--regional_seismic_stations",
        type=str,
        default="auto",
        help="""
        regional seismic stations for validating the models
        - 'auto': would be automatically determined.
        - OR: list of coma separated station, e.g.
        "NC.KRP,BK.SBAR,BK.THOM,BK.DMOR,NC.KMPB,BK.HUNT,BK.ETSL,BK.HALS,BK.MNDO"
        """,
    )

    parser.add_argument(
        "--teleseismic_stations",
        type=str,
        default="auto",
        help="""
        teleseismic seismic stations for validating the models
        - 'auto': would be automatically determined.
        - OR: list of coma separated station, e.g.
        "NC.KRP,BK.SBAR,BK.THOM,BK.DMOR,NC.KMPB,BK.HUNT,BK.ETSL,BK.HALS,BK.MNDO"
        """,
    )

    parser.add_argument(
        "--regional_synthetics_generator",
        type=str,
        choices=["axitra", "seissol"],
        default="axitra",
        help="""
            Tool used for generating regional waveform synthetics.
        """,
    )

    parser.add_argument(
        "--first_simulation_id",
        type=int,
        default=0,
        help="""
        first simulation id to be use in the ensemble
        """,
    )

    parser.add_argument(
        "--seissol_end_time",
        type=str,
        default="auto",
        help="""
        End time for SeisSol simulation.
        Options:
        - 'auto': Automatically determined based on estimated earthquake duration.
        - A float value (in seconds): Specify a manual end time.
        """,
    )

    parser.add_argument(
        "--template_folder",
        type=str,
        default=None,
        help="Path to a folder with customized fault or parameter templates",
    )

    parser.add_argument(
        "--terminator",
        type=str,
        default="auto",
        help="""
        Controls whether the SeisSol terminator is enabled.
        Options:
        - 'auto': Enabled if --seissol_end_time is 'auto', otherwise disabled.
        - 'True' or 'False': Manually enable or disable the terminator.
        """,
    )

    parser.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="""
        Maximum rupture time in seconds.
        Slip contributions with t_rupt > tmax will be ignored.
        """,
    )

    parser.add_argument(
        "--velocity_model",
        type=str,
        default="auto",
        help="""
        Velocity model to use.
        - 'auto': choose based on finite fault model (e.g., Slipnear or USGS).
        - 'usgs': extract from the USGS FSP file.
        - OR: provide a velocity model in Axitra format.
        """,
    )

    return parser


def get_args(argv=None):
    """
    Parse command-line arguments for the workflow.

    The test on argv is required because pytest adds its own arguments
    when running tests, which would otherwise cause parsing to fail.
    """
    parser = get_parser()
    if argv is None:
        argv = sys.argv[1:]
    return parser.parse_known_args(argv)[0]
