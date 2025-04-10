from fault_plane import FaultPlane
import os
import numpy as np


class MultiFaultPlane:
    def __init__(self, fault_planes, hypocenter=None):
        self.fault_planes = fault_planes
        self.max_slip = 0.0
        self.hypocenter = hypocenter
        for fp in fault_planes:
            self.max_slip = max(self.max_slip, np.amax(fp.slip1))

    @classmethod
    def from_file(cls, filename):
        """Load a MultiFaultPlane from a file based on its extension."""
        ext = os.path.splitext(filename)[1].lower()  # Ensure lowercase extensions

        if ext == ".srf":
            mfp = cls.from_srf(filename)
        elif ext == ".param":
            mfp = cls.from_usgs_param_file(filename)
        elif ext == ".param2":
            mfp = cls.from_usgs_param_file_alternative(filename)
        elif ext == ".fsp":
            mfp = cls.from_usgs_fsp_file(filename)
        elif ext == ".txt":
            mfp = cls.from_slipnear_param_file(filename)
        else:
            raise NotImplementedError(f"Unknown extension: {ext}")
        if mfp.is_static_solution():
            mfp.hypocenter = None
        return mfp

    @classmethod
    def from_usgs_fsp_file(cls, fname):
        import re
        import pandas as pd
        from io import StringIO

        with open(fname, "r") as fid:
            lines = fid.readlines()

        if "FINITE-SOURCE RUPTURE MODEL" not in lines[0]:
            raise ValueError("Not a valid USGS fsp file.")

        def read_param(line, name, dtype=int):
            if name not in line:
                raise ValueError(f"{name} not found in line: {line}")
            else:
                return dtype(line.split(f"{name} =")[1].split()[0])

        def get_to_first_line_starting_with(lines, pattern):
            for i, line in enumerate(lines):
                if line.startswith(pattern):
                    return lines[i:]
            raise ValueError(f"{pattern} not found")

        lines = get_to_first_line_starting_with(lines, "% Mech :")
        strike = read_param(lines[0], "STRK", float)
        dip = read_param(lines[0], "DIP", float)
        lines = get_to_first_line_starting_with(lines, "% Invs :")
        nx = read_param(lines[0], "Nx")
        ny = read_param(lines[0], "Nz")
        dx = read_param(lines[1], "Dx", float)
        dy = read_param(lines[1], "Dz", float)
        nseg = read_param(lines[2], "Nsg")
        print(f"No. of fault segments in param file: {nseg}")

        lines = get_to_first_line_starting_with(lines, "% VELOCITY-DENSITY")
        nlayers = read_param(lines[1], "layers")
        text_file = StringIO("\n".join(lines[3 : 5 + nlayers]))
        velocity_model_df = pd.read_csv(text_file, sep=r"\s+").drop(0)
        print(velocity_model_df)

        fault_planes = []
        for i_seg in range(nseg):
            fault_planes.append(FaultPlane())
            fp = fault_planes[i_seg]
            if nseg != 1:
                lines = get_to_first_line_starting_with(lines, "% SEGMENT")
                strike = read_param(lines[0], "STRIKE", float)
                dip = read_param(lines[0], "DIP", float)
                lx = read_param(lines[0], "LEN", float)
                ly = read_param(lines[0], "WID", float)
                nx = round(lx / dx)
                ny = round(ly / dy)

            lines = get_to_first_line_starting_with(lines, "% Nsbfs")
            nsbfs = read_param(lines[0], "Nsbfs")
            assert nsbfs == nx * ny

            fp.dx = dx
            fp.dy = dy
            fp.init_spatial_arrays(nx, ny)
            lines = get_to_first_line_starting_with(lines, "% LAT LON")
            column_names = lines[0][1:].split()
            text_file = StringIO("\n".join(lines[2 : 2 + nsbfs]))
            df = pd.read_csv(text_file, sep=r"\s+", header=None, names=column_names)

            assert (
                df["TRUP"] >= 0
            ).all(), (
                "AssertionError: Not all rupture time are greater than or equal to 0."
            )
            for j in range(fp.ny):
                for i in range(fp.nx):
                    k = j * fp.nx + i
                    fp.lon[j, i] = df["LON"][k]
                    fp.lat[j, i] = df["LAT"][k]
                    fp.depth[j, i] = df["Z"][k]
                    fp.slip1[j, i] = df["SLIP"][k]
                    fp.rake[j, i] = df["RAKE"][k]
                    fp.strike[j, i] = strike
                    fp.dip[j, i] = dip
                    fp.PSarea_cm2 = dx * dy * 1e10
                    fp.t0[j, i] = df["TRUP"][k]
                    # t_fal in not specified in this file (compared with the *.param file)
                    fp.tacc[j, i] = 0.5 * df["RISE"][k]
                    fp.rise_time[j, i] = df["RISE"][k]
        return cls(fault_planes)

    @classmethod
    def from_usgs_param_file_alternative(cls, fname):
        # format using in Xu et al. (2024), Noto earthquake
        import re
        import pandas as pd
        from io import StringIO

        header = "lat lon depth slip rake strike dip t_rup t_ris t_fal mo"
        with open(fname, "r") as fid:
            lines = fid.readlines()

        nseg_line = [line for line in lines if "#Total number of rectangular" in line]
        if len(nseg_line) != 1:
            raise ValueError("Not a valid USGS2 param file.")

        nseg = int(nseg_line[0].split()[-1])  # number of fault segments
        print(f"No. of fault segments in param file: {nseg}")

        fault_seg_line = [line for line in lines if "#NX= " in line]
        assert (
            len(fault_seg_line) == nseg
        ), f"No. of segments are wrong. {len(fault_seg_line)} {nseg}"
        istart = 4
        fault_planes = []
        t0min = 1e99
        for i_seg in range(nseg):
            fault_planes.append(FaultPlane())
            fp = fault_planes[i_seg]

            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", fault_seg_line[i_seg])
            # Convert extracted strings to float
            numbers = [float(num) for num in numbers]
            nx, dx, ny, dy = numbers
            nx, ny = map(int, [nx, ny])
            fp.dx = dx
            fp.dy = dy
            fp.init_spatial_arrays(nx, ny)

            line1 = istart + 15
            line2 = line1 + nx * ny
            istart = line1 + nx * ny
            text_file = StringIO("\n".join([header, *lines[line1:line2]]))
            df = pd.read_csv(text_file, sep=r"\s+")

            assert (
                df["t_rup"] >= 0
            ).all(), (
                "AssertionError: Not all rupture time are greater than or equal to 0."
            )
            for j in range(fp.ny):
                for i in range(fp.nx):
                    k = j * fp.nx + i
                    fp.lon[j, i] = df["lon"][k]
                    fp.lat[j, i] = df["lat"][k]
                    fp.depth[j, i] = df["depth"][k]
                    fp.slip1[j, i] = df["slip"][k]
                    fp.rake[j, i] = df["rake"][k]
                    fp.strike[j, i] = df["strike"][k]
                    fp.dip[j, i] = df["dip"][k]
                    fp.PSarea_cm2 = dx * dy * 1e10
                    fp.t0[j, i] = df["t_rup"][k]
                    fp.tacc[j, i] = df["t_ris"][k]
                    fp.rise_time[j, i] = df["t_ris"][k] + df["t_fal"][k]
            if np.amin(fp.t0[:, :]) < t0min:
                t0min = np.amin(fp.t0[:, :])
                ids = np.where(fp.t0[:, :] == t0min)
                if len(ids[0]) > 1:
                    print(ids)
                    raise ValueError("more than one hypocenter?")
                else:
                    hypocenter = [*fp.lon[ids], *fp.lat[ids], *fp.depth[ids]]

        return cls(fault_planes, hypocenter)

    @classmethod
    def from_usgs_param_file(cls, fname):
        import re
        import pandas as pd
        from io import StringIO

        header = "lat lon depth slip rake strike dip t_rup t_ris t_fal mo"
        with open(fname, "r") as fid:
            lines = fid.readlines()

        if "#Total number of fault_segments" not in lines[0]:
            raise ValueError("Not a valid USGS param file.")

        nseg = int(lines[0].split()[-1])  # number of fault segments
        print(f"No. of fault segments in param file: {nseg}")

        fault_seg_line = [line for line in lines if "#Fault_segment " in line]
        assert (
            len(fault_seg_line) == nseg
        ), f"No. of segments are wrong. {len(fault_seg_line)} {nseg}"

        istart = 1
        fault_planes = []
        t0min = 1e99
        for i_seg in range(nseg):
            fault_planes.append(FaultPlane())
            fp = fault_planes[i_seg]

            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", fault_seg_line[i_seg])

            # Convert extracted strings to float
            numbers = [float(num) for num in numbers]
            _, nx, dx, ny, dy = numbers
            nx, ny = map(int, [nx, ny])
            fp.dx = dx
            fp.dy = dy
            fp.init_spatial_arrays(nx, ny)

            line1 = istart + 9
            line2 = line1 + nx * ny
            istart = line1 + nx * ny

            text_file = StringIO("\n".join([header, *lines[line1:line2]]))
            df = pd.read_csv(text_file, sep=r"\s+")

            assert (
                df["t_rup"] >= 0
            ).all(), (
                "AssertionError: Not all rupture time are greater than or equal to 0."
            )
            for j in range(fp.ny):
                for i in range(fp.nx):
                    k = j * fp.nx + i
                    fp.lon[j, i] = df["lon"][k]
                    fp.lat[j, i] = df["lat"][k]
                    fp.depth[j, i] = df["depth"][k]
                    fp.slip1[j, i] = df["slip"][k]
                    fp.rake[j, i] = df["rake"][k]
                    fp.strike[j, i] = df["strike"][k]
                    fp.dip[j, i] = df["dip"][k]
                    fp.PSarea_cm2 = dx * dy * 1e10
                    fp.t0[j, i] = df["t_rup"][k]
                    fp.tacc[j, i] = df["t_ris"][k]
                    fp.rise_time[j, i] = df["t_ris"][k] + df["t_fal"][k]

            if np.amin(fp.t0[:, :]) < t0min:
                t0min = np.amin(fp.t0[:, :])
                ids = np.where(fp.t0[:, :] == t0min)
                hypocenter = [*fp.lon[ids], *fp.lat[ids], *fp.depth[ids]]

        if nseg == 1:
            fault_planes[0] = fp.trim()

        return cls(fault_planes, hypocenter)

    @classmethod
    def from_slipnear_param_file(cls, fname):
        import pandas as pd

        """ reading a file from the SLIPNEAR method, Delouis, Géoazur/OCA """

        def read_dx_dy_hypocenter(fname):
            with open(fname, "r") as fid:
                lines = fid.readlines()

            if not lines[0].startswith(" RECTANGULAR DISLOCATION MODEL"):
                raise ValueError("Not a valid slipnear param file.")

            dx = float(lines[2].split()[3])
            dy = float(lines[3].split()[0])
            hypocenter = None
            for line in lines:
                if line.strip().endswith("hypocenter"):
                    hypocenter = [float(v) for v in line.split()[0:3]]
                    break
            if not hypocenter:
                raise ValueError("Failed reading hypocenter from slipnear file")
            hypocenter[0], hypocenter[1] = hypocenter[1], hypocenter[0]
            return dx, dy, hypocenter

        print(f"reading {fname}, assuming it is a slipnear file")
        dx, dy, hypocenter = read_dx_dy_hypocenter(fname)
        fault_planes = []
        fault_planes.append(FaultPlane())
        fp = fault_planes[0]

        def get_first_line_id_starting_with(fname, pattern):
            with open(fname, "r") as fid:
                lines = fid.readlines()
            for i, line in enumerate(lines):
                if line.startswith(pattern):
                    return i
            raise ValueError(f"{pattern} not found")

        try:
            with open(fname, "r") as fid:
                lines = fid.readlines()
            line_number_of_triangular = get_first_line_id_starting_with(
                fname, " per subfault:"
            )
            ntriangles = int(lines[line_number_of_triangular].split(":")[1])
            line_half_dur = get_first_line_id_starting_with(
                fname, " isoceles triangular functions (s):"
            )
            half_dur = float(lines[line_half_dur].split(":")[1])
            rise_time = half_dur * (1 + ntriangles)
            tacc = half_dur
            print(f"rise time and tacc inferred:  {tacc} {rise_time}")
            has_STF = True
        except ValueError:
            rise_time = 10.0
            tacc = 5.0
            has_STF = False
            print(
                f"rise time and tacc could not be determined, using {tacc} {rise_time}"
            )

        line_sep = get_first_line_id_starting_with(
            fname, " ================================================"
        )

        df = pd.read_csv(fname, sep=r"\s+", skiprows=line_sep + 2, comment=":")
        df = df.sort_values(
            by=["depth(km)", "Lat", "Lon"], ascending=[True, True, True]
        ).reset_index()
        rows_with_same_depth = df[df["depth(km)"] == df.iloc[0]["depth(km)"]]
        nx = len(rows_with_same_depth)
        ny = len(df) // nx
        assert len(df) % nx == 0

        print(f"read {nx} x {ny} fault segments of size {dx} x {dy} km2 in param file")
        print(df)

        fp.dx = dx
        fp.dy = dy
        fp.init_spatial_arrays(nx, ny)

        def G(depth):
            if depth < 0.6:
                return 7.2200000000e09
            elif depth < 2:
                return 1.5548000000e10
            elif depth < 5:
                return 2.5281000000e10
            elif depth < 30:
                return 4.0781250000e10
            else:
                return 7.2277920000e10

        assert (
            df["ontime"] >= 0
        ).all(), "AssertionError: Not all rupture time are greater than or equal to 0."
        Gslip = 0
        Gslip25 = 0
        for j in range(fp.ny):
            for i in range(fp.nx):
                k = j * fp.nx + i
                fp.lon[j, i] = df["Lon"][k]
                fp.lat[j, i] = df["Lat"][k]
                fp.depth[j, i] = df["depth(km)"][k]
                # the slip is based on a homogeneous mu model, but the waveforms
                # are calculated with the layered G as above
                fp.slip1[j, i] = df["slip(cm)"][k]  # * 2.5e10/ G(fp.depth[j, i])
                Gslip25 += df["slip(cm)"][k] * 2.5e10
                Gslip += df["slip(cm)"][k] * G(fp.depth[j, i])
                fp.rake[j, i] = df["rake"][k]
                fp.strike[j, i] = df["strike"][k]
                fp.dip[j, i] = df["dip"][k]
                fp.PSarea_cm2 = dx * dy * 1e10
                fp.t0[j, i] = df["ontime"][k]
                fp.tacc[j, i] = tacc
                fp.rise_time[j, i] = rise_time
                if has_STF:
                    dt = tacc / 25.0
                    ndt1 = int((fp.t0[j, i] + fp.rise_time[j, i]) / dt) + 1
                    if max(i, j) == 0:
                        fp.ndt = ndt1
                        fp.dt = dt
                        fp.init_aSR()
                        fp.myt = np.linspace(0, fp.ndt - 1, fp.ndt) * fp.dt
                    if ndt1 == 0:
                        continue
                    if ndt1 > fp.ndt:
                        print(
                            f"a larger ndt ({ndt1}> {fp.ndt}) was found for point source (i,j) = ({i}, {j}) extending aSR array..."
                        )
                        fp.extend_aSR(fp.ndt, ndt1)
                        fp.myt = np.linspace(0, fp.ndt - 1, fp.ndt) * fp.dt

                    # Function to generate a single triangle function
                    def triangle_function(t, center, half_width, amplitude):
                        return np.where(
                            np.abs(t - center) <= half_width,
                            amplitude * (1 - np.abs(t - center) / half_width),
                            0,
                        )

                    # Generate the resulting signal as a sum of multiple triangle functions
                    def sum_of_triangles(t, triangles):
                        signal = np.zeros_like(t)
                        for center, half_width, amplitude in triangles:
                            signal += triangle_function(
                                t, center, half_width, amplitude
                            )
                        return signal

                    triangles = []
                    for itr in range(ntriangles):
                        triangles.append(
                            [
                                fp.t0[j, i] + (itr + 1) * tacc,
                                tacc,
                                df[f"amp{itr + 1}(dyne.cm/s)"][k],
                            ]
                        )
                    fp.aSR[j, i, :] = sum_of_triangles(fp.myt, triangles)
                    integral = np.trapz(fp.aSR[j, i, :], dx=fp.dt)
                    if integral:
                        fp.aSR[j, i, :] /= integral
        # we scale down the model to have the expected magnitude (see comment above on 25GPa)
        fp.slip1 *= Gslip25 / Gslip
        fault_planes[0] = fp.trim()
        fault_planes[0] = fault_planes[0].add_one_zero_slip_row_at_depth()
        return cls(fault_planes, hypocenter)

    @classmethod
    def from_srf(cls, fname):
        "init object by reading a srf file (standard rutpure format)"
        fault_planes = []
        with open(fname) as fid:
            # version
            line = fid.readline()
            if line.strip() not in ["1.0", "2.0"]:
                raise NotImplementedError(f"srf version: {line} not supported")
            # skip comments
            while True:
                line = fid.readline()
                if not line.startswith("#"):
                    break
            line_el = line.split()
            if line_el[0] != "PLANE":
                raise ValueError(
                    f"error parsing {fname}: line does not start with PLANE : {line}"
                )
            nplane = int(line_el[1])
            for p in range(nplane):
                line_el = fid.readline().split()
                nx, ny = [int(val) for val in line_el[2:4]]
                fault_planes.append(FaultPlane())
                fault_planes[p].init_spatial_arrays(nx, ny)
                if len(line_el) == 6:
                    # the line describing the fault plane is divided in 2
                    fid.readline()
            for p in range(nplane):
                fp = fault_planes[p]
                print(f"processing fault plane {p}, {fp.nx} {fp.ny}")
                line_el = fid.readline().split()
                if line_el[0] != "POINTS":
                    raise ValueError(
                        f"error parsing {fname}: line does not start with POINTS : {line}"
                    )
                # check that the plane data are consistent with the number of points
                assert int(line_el[1]) == fp.nx * fp.ny
                for j in range(fp.ny):
                    for i in range(fp.nx):
                        # first header line
                        line = fid.readline()
                        # rho_vs are only present for srf version 2
                        (
                            fp.lon[j, i],
                            fp.lat[j, i],
                            fp.depth[j, i],
                            fp.strike[j, i],
                            fp.dip[j, i],
                            fp.PSarea_cm2,
                            fp.t0[j, i],
                            dt,
                            *rho_vs,
                        ) = [float(v) for v in line.split()]
                        # second header line
                        line = fid.readline()
                        (
                            fp.rake[j, i],
                            fp.slip1[j, i],
                            ndt1,
                            slip2,
                            ndt2,
                            slip3,
                            ndt3,
                        ) = [float(v) for v in line.split()]
                        if max(slip2, slip3) > 0.0:
                            raise NotImplementedError(
                                "this script assumes slip2 and slip3 are zero",
                                slip2,
                                slip3,
                            )
                        ndt1 = int(ndt1)
                        if max(i, j) == 0:
                            fp.ndt = ndt1
                            fp.dt = dt
                            fp.init_aSR()
                        lSTF = []
                        if ndt1 == 0:
                            continue
                        if ndt1 > fp.ndt:
                            print(
                                f"a larger ndt ({ndt1}> {fp.ndt}) was found for point source (i,j) = ({i}, {j}) extending aSR array..."
                            )
                            fp.extend_aSR(fp.ndt, ndt1)
                        if abs(dt - fp.dt) > 1e-6:
                            raise NotImplementedError(
                                "this script assumes that dt is the same for all sources",
                                dt,
                                fp.dt,
                            )
                        while True:
                            line = fid.readline()
                            lSTF.extend(line.split())
                            if len(lSTF) == ndt1:
                                fp.aSR[j, i, 0:ndt1] = np.array(
                                    [float(v) for v in lSTF]
                                )
                                break
            return cls(fault_planes)

    def temporal_crop(self, tmax):
        """remove fault slip for t_rupt> tmax (slip fitting waveform noise?)"""
        print(f"croping finite fault model, removing slip at t>{tmax}")
        for p, fp in enumerate(self.fault_planes):
            ids = np.where(fp.t0 > tmax)
            if len(ids) > 0:
                fp.slip1[ids] = 0.01

    def is_static_solution(self):
        """check if t0==0 for all sources"""
        t0max = 0.0
        for p, fp in enumerate(self.fault_planes):
            t0max = max(t0max, np.amax(fp.t0[:, :]))
        return t0max == 0.0

    def generate_fault_ts_yaml_fl33(self, prefix, method, spatial_zoom, proj):
        """Generate yaml file initializing FL33 arrays and ts file describing the planar fault geometry."""

        if not os.path.exists("yaml_files"):
            os.makedirs("yaml_files")
        # Generate yaml file loading ASAGI file
        template_yaml = """!Switch
[strike_slip, dip_slip, rupture_onset, tau_S, tau_R, rupture_rise_time, rake_interp_low_slip]: !EvalModel
    parameters: [strike_slip, dip_slip, rupture_onset, effective_rise_time, acc_time, rake_interp_low_slip]
    model: !Any
     components:
"""
        for p, fp in enumerate(self.fault_planes):
            fault_id = 3 if p == 0 else 64 + p
            fp.compute_xy_from_latlon(proj)
            fp.compute_affine_vector_map()
            hh = fp.affine_map["hh"]
            hw = fp.affine_map["hw"]
            t1 = fp.affine_map["t1"]
            t2 = fp.affine_map["t2"]
            fp.write_ts_file(f"{prefix}{fault_id}")

            template_yaml += f"""      - !GroupFilter
        groups: {fault_id}
        components: !AffineMap
              matrix:
                ua: [{hh[0]}, {hh[1]}, {hh[2]}]
                ub: [{hw[0]}, {hw[1]}, {hw[2]}]
              translation:
                ua: {t1}
                ub: {t2}
              components: !Any
                - !ASAGI
                    file: ASAGI_files/{prefix}{p + 1}_{spatial_zoom}_{method}.nc
                    parameters: [strike_slip, dip_slip, rupture_onset, effective_rise_time, acc_time, rake_interp_low_slip]
                    var: data
                    interpolation: linear
                - !ConstantMap
                  map:
                    strike_slip: 0.0
                    dip_slip:    0.0
                    rupture_onset:    0.0
                    acc_time:  1e100
                    effective_rise_time:  2e100
                    rake_interp_low_slip: 0.0
"""
        template_yaml += """    components: !LuaMap
      returns: [strike_slip, dip_slip, rupture_onset, tau_S, tau_R, rupture_rise_time, rake_interp_low_slip]
      function: |
        function f (x)
          -- Note the minus on strike_slip to acknowledge the different convention of SeisSol (T_s>0 means right-lateral)
          -- same for the math.pi factor on rake
          return {
          strike_slip = -x["strike_slip"],
          dip_slip = x["dip_slip"],
          rupture_onset = x["rupture_onset"],
          tau_S = x["acc_time"]/1.27,
          tau_R = x["effective_rise_time"] - 2.*x["acc_time"]/1.27,
          rupture_rise_time = x["effective_rise_time"],
          rake_interp_low_slip = math.pi - x["rake_interp_low_slip"]
          }
        end
        """

        fname = "yaml_files/FL33_34_fault.yaml"
        with open(fname, "w") as fid:
            fid.write(template_yaml)
        print(f"done writing {fname}")
        if self.hypocenter:
            if not os.path.exists("tmp"):
                os.makedirs("tmp")
            with open("tmp/hypocenter.txt", "w") as f:
                f.write(
                    f"{self.hypocenter[0]} {self.hypocenter[1]} {self.hypocenter[2]}\n"
                )
