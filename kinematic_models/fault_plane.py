# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

import os

import numpy as np
import scipy.ndimage
import xarray as xr
from asagiwriter import writeNetcdf
from scipy import interpolate, ndimage
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import gaussian_filter
from stf import gaussianSTF, regularizedYoffe


def cosine_taper(npts, p=0.1, freqs=None, flimit=None, halfcosine=True, sactaper=False):
    """
    Cosine Taper. (copied from obspy:
    https://docs.obspy.org/master/_modules/obspy/signal/invsim.html#cosine_taper

    :type npts: int
    :param npts: Number of points of cosine taper.
    :type p: float
    :param p: Decimal percentage of cosine taper (ranging from 0 to 1). Default
        is 0.1 (10%) which tapers 5% from the beginning and 5% form the end.
    :rtype: float NumPy :class:`~numpy.ndarray`
    :return: Cosine taper array/vector of length npts.
    :type freqs: NumPy :class:`~numpy.ndarray`
    :param freqs: Frequencies as, for example, returned by fftfreq
    :type flimit: list or tuple of floats
    :param flimit: The list or tuple defines the four corner frequencies
        (f1, f2, f3, f4) of the cosine taper which is one between f2 and f3 and
        tapers to zero for f1 < f < f2 and f3 < f < f4.
    :type halfcosine: bool
    :param halfcosine: If True the taper is a half cosine function. If False it
        is a quarter cosine function.
    :type sactaper: bool
    :param sactaper: If set to True the cosine taper already tapers at the
        corner frequency (SAC behavior). By default, the taper has a value
        of 1.0 at the corner frequencies.

    .. rubric:: Example

    >>> tap = cosine_taper(100, 1.0)
    >>> tap2 = 0.5 * (1 + np.cos(np.linspace(np.pi, 2 * np.pi, 50)))
    >>> np.allclose(tap[0:50], tap2)
    True
    >>> npts = 100
    >>> p = 0.1
    >>> tap3 = cosine_taper(npts, p)
    >>> (tap3[int(npts*p/2):int(npts*(1-p/2))]==np.ones(int(npts*(1-p)))).all()
    True
    """
    if p < 0 or p > 1:
        msg = "Decimal taper percentage must be between 0 and 1."
        raise ValueError(msg)
    if p == 0.0 or p == 1.0:
        frac = int(npts * p / 2.0)
    else:
        frac = int(npts * p / 2.0 + 0.5)

    if freqs is not None and flimit is not None:
        fl1, fl2, fl3, fl4 = flimit
        idx1 = np.argmin(abs(freqs - fl1))
        idx2 = np.argmin(abs(freqs - fl2))
        idx3 = np.argmin(abs(freqs - fl3))
        idx4 = np.argmin(abs(freqs - fl4))
    else:
        idx1 = 0
        idx2 = frac - 1
        idx3 = npts - frac
        idx4 = npts - 1
    if sactaper:
        # in SAC the second and third
        # index are already tapered
        idx2 += 1
        idx3 -= 1

    # Very small data lengths or small decimal taper percentages can result in
    # idx1 == idx2 and idx3 == idx4. This breaks the following calculations.
    if idx1 == idx2:
        idx2 += 1
    if idx3 == idx4:
        idx3 -= 1

    # the taper at idx1 and idx4 equals zero and
    # at idx2 and idx3 equals one
    cos_win = np.zeros(npts)
    if halfcosine:
        # cos_win[idx1:idx2+1] =  0.5 * (1.0 + np.cos((np.pi * \
        #    (idx2 - np.arange(idx1, idx2+1)) / (idx2 - idx1))))
        cos_win[idx1 : idx2 + 1] = 0.5 * (
            1.0
            - np.cos(
                (np.pi * (np.arange(idx1, idx2 + 1) - float(idx1)) / (idx2 - idx1))
            )
        )
        cos_win[idx2 + 1 : idx3] = 1.0
        cos_win[idx3 : idx4 + 1] = 0.5 * (
            1.0
            + np.cos(
                (np.pi * (float(idx3) - np.arange(idx3, idx4 + 1)) / (idx4 - idx3))
            )
        )
    else:
        cos_win[idx1 : idx2 + 1] = np.cos(
            -(np.pi / 2.0 * (float(idx2) - np.arange(idx1, idx2 + 1)) / (idx2 - idx1))
        )
        cos_win[idx2 + 1 : idx3] = 1.0
        cos_win[idx3 : idx4 + 1] = np.cos(
            (np.pi / 2.0 * (float(idx3) - np.arange(idx3, idx4 + 1)) / (idx4 - idx3))
        )

    # if indices are identical division by zero
    # causes NaN values in cos_win
    if idx1 == idx2:
        cos_win[idx1] = 0.0
    if idx3 == idx4:
        cos_win[idx3] = 0.0
    return cos_win


def interpolate_nan_from_neighbors(array):
    """rise_time and tacc may not be defined where there is no slip (no SR function).
    in this case, we interpolate from neighbors
    source:
    https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python
    """
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    # mask invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]
    return interpolate.griddata(
        (x1, y1),
        newarr.ravel(),
        (xx, yy),
        method="linear",
        fill_value=np.average(array),
    )


def compute_block_mean(ar, fact):
    a = xr.DataArray(ar, dims=["x", "y"])
    return a.coarsen(x=fact, y=fact).mean().to_numpy()


def upsample_quantities(
    allarr,
    spatial_order,
    spatial_zoom,
    padding="constant",
    extra_padding_layer=False,
    minimize_block_average_variations=False,
):
    """1. pad
    2. upsample, adding spatial_zoom per node
    """
    nd = allarr.shape[0]
    ny, nx = [val * spatial_zoom for val in allarr[0].shape]
    if extra_padding_layer:
        # required for vertex aligned netcdf format
        nx = nx + 2
        ny = ny + 2
    allarr0 = np.zeros((nd, ny, nx))
    for k in range(nd):
        if padding == "extrapolate":
            my_array0 = np.pad(
                allarr[k, :, :], ((1, 1), (1, 1)), "reflect", reflect_type="odd"
            )
        else:
            my_array0 = np.pad(allarr[k, :, :], ((1, 1), (1, 1)), padding)
        if extra_padding_layer:
            ncrop = spatial_zoom - 1
        else:
            ncrop = spatial_zoom
        my_array = scipy.ndimage.zoom(
            my_array0,
            spatial_zoom,
            order=spatial_order,
            mode="grid-constant",
            grid_mode=True,
        )
        if minimize_block_average_variations:
            # inspired by Tinti et al. (2005) (Appendix A)
            # This is for the specific case of fault slip.
            # We want to preserve the seismic moment of each subfault after
            # interpolation
            # the rock rigidity is not know by this script (would require some python
            # binding of easi).
            # the subfault area is typically constant over the kinematic model
            # So we just want to perserve subfault average.
            print("trying to perserve subfault average...")
            my_array = np.maximum(0, my_array)
            best_misfit = float("inf")
            # The algorithm does not seem to converge, but produces better model
            # (given the misfit) that inital after 2-3 iterations
            niter = 30
            for i in range(niter):
                block_average = compute_block_mean(my_array, spatial_zoom)
                correction = my_array0 / block_average
                # having a misfit as misfit = np.linalg.norm(correction) does not
                # makes sense as for almost 0 slip, correction can be large
                misfit = np.linalg.norm(my_array0 - block_average) / len(my_array0)
                if best_misfit > misfit:
                    if i == 0:
                        print(f"misfit at iter {i}: {misfit}")
                    else:
                        print(f"misfit improved at iter {i}: {misfit}")
                    best_misfit = misfit
                    best = np.copy(my_array)
                my_array = scipy.ndimage.zoom(
                    correction * my_array0,
                    spatial_zoom,
                    order=spatial_order,
                    mode="grid-constant",
                    grid_mode=True,
                )
                my_array = np.maximum(0, my_array)
            my_array = best
        if ncrop > 0:
            allarr0[k, :, :] = my_array[ncrop:-ncrop, ncrop:-ncrop]
        else:
            allarr0[k, :, :] = my_array

    return allarr0


class FaultPlane:
    def __init__(self):
        self.nx = 0
        self.ny = 0
        self.dx = None
        self.dy = None
        self.ndt = 0
        self.PSarea_cm2 = 0
        self.dt = 0
        # array member initialized to dummy value
        self.lon = 0
        self.lat = 0
        self.x = 0
        self.y = 0
        self.depth = 0
        self.t0 = 0
        self.slip1 = 0
        self.strike = 0
        self.dip = 0
        self.rake = 0
        self.aSR = 0
        self.myt = 0

    def init_spatial_arrays(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.lon = np.zeros((ny, nx))
        self.lat = np.zeros((ny, nx))
        self.x = np.zeros((ny, nx))
        self.y = np.zeros((ny, nx))
        self.depth = np.zeros((ny, nx))
        self.t0 = np.zeros((ny, nx))
        self.slip1 = np.zeros((ny, nx))
        self.strike = np.zeros((ny, nx))
        self.dip = np.zeros((ny, nx))
        self.rake = np.zeros((ny, nx))
        self.rise_time = np.zeros((self.ny, self.nx))
        self.tacc = np.zeros((self.ny, self.nx))

    def init_aSR(self):
        self.aSR = np.zeros((self.ny, self.nx, self.ndt))

    def extend_aSR(self, ndt_old, ndt_new):
        "extend aSR array to more time samplings"
        tmpSR = np.copy(self.aSR)
        self.ndt = ndt_new
        self.aSR = np.zeros((self.ny, self.nx, self.ndt))
        self.aSR[:, :, 0:ndt_old] = tmpSR[:, :, :]

    def compute_xy_from_latlon(self, proj):
        if proj:
            from pyproj import Transformer

            transformer = Transformer.from_crs("epsg:4326", proj, always_xy=True)
            self.x, self.y = transformer.transform(self.lon, self.lat)
        else:
            print("no proj string specified!")
            self.x, self.y = self.lon, self.lat

    def compute_latlon_from_xy(self, proj):
        if proj:
            from pyproj import Transformer

            transformer = Transformer.from_crs(proj, "epsg:4326", always_xy=True)
            self.lon, self.lat = transformer.transform(self.x, self.y)
        else:
            self.lon, self.lat = self.x, self.y

    def compute_time_array(self):
        self.myt = np.linspace(0, (self.ndt - 1) * self.dt, self.ndt)

    def write_srf(self, fname):
        "write kinematic model to a srf file (standard rutpure format)"
        with open(fname, "w") as fout:
            fout.write("1.0\n")
            fout.write("POINTS %d\n" % (self.nx * self.ny))
            for j in range(self.ny):
                for i in range(self.nx):
                    fout.write(
                        "%g %g %g %g %g %e %g %g\n"
                        % (
                            self.lon[j, i],
                            self.lat[j, i],
                            self.depth[j, i],
                            self.strike[j, i],
                            self.dip[j, i],
                            self.PSarea_cm2,
                            self.t0[j, i],
                            self.dt,
                        )
                    )
                    fout.write(
                        "%g %g %d %f %d %f %d\n"
                        % (self.rake[j, i], self.slip1[j, i], self.ndt, 0.0, 0, 0.0, 0)
                    )
                    np.savetxt(fout, self.aSR[j, i, :], fmt="%g", newline=" ")
                    fout.write("\n")
        print("done writing", fname)

    def assess_STF_parameters(self, threshold):
        "compute rise_time (slip duration) and t_acc (peak SR) from SR time histories"
        assert threshold >= 0.0 and threshold < 1
        misfits_Yoffe = []
        misfits_Gaussian = []
        for j in range(self.ny):
            for i in range(self.nx):
                if not self.slip1[j, i]:
                    self.rise_time[j, i] = np.nan
                    self.tacc[j, i] = np.nan
                else:
                    peakSR = np.amax(self.aSR[j, i, :])
                    id_max = np.where(self.aSR[j, i, :] == peakSR)[0][0]
                    ids_greater_than_threshold = np.where(
                        self.aSR[j, i, :] > threshold * peakSR
                    )[0]
                    first_non_zero = np.amin(ids_greater_than_threshold)
                    last_non_zero = np.amax(ids_greater_than_threshold)
                    self.rise_time[j, i] = (
                        last_non_zero - first_non_zero + 1
                    ) * self.dt
                    self.tacc[j, i] = (id_max - first_non_zero + 1) * self.dt
                    t0_increment = first_non_zero * self.dt
                    self.t0[j, i] += t0_increment
                    # 2 dims: 0: Yoffe 1: Gaussian
                    newSR = np.zeros((self.ndt, 2))
                    # Ts and Td parameters of the Yoffe function have no direct physical
                    # meaning
                    # Tinti et al. (2005) suggest that Ts can nevertheless be associated
                    # with the acceleration time tacc Empirically, they find that Ts and
                    # Tacc are for most (Ts,Td) parameter sets linearly related
                    # with the 'magic' number 1.27
                    ts = self.tacc[j, i] / 1.27
                    tr = self.rise_time[j, i] - 2.0 * ts
                    tr = max(tr, ts)
                    for k, tk in enumerate(self.myt):
                        newSR[k, 0] = regularizedYoffe(tk - t0_increment, ts, tr)
                        newSR[k, 1] = gaussianSTF(
                            tk - t0_increment, self.rise_time[j, i], self.dt
                        )
                    integral_aSTF = np.trapz(np.abs(self.aSR[j, i, :]), dx=self.dt)
                    integral_Yoffe = np.trapz(np.abs(newSR[:, 0]), dx=self.dt)
                    integral_Gaussian = np.trapz(np.abs(newSR[:, 1]), dx=self.dt)
                    if integral_aSTF > 0:
                        misfits_Yoffe.append(
                            np.linalg.norm(
                                self.aSR[j, i, :] / integral_aSTF
                                - newSR[:, 0] / integral_Yoffe
                            )
                        )
                        misfits_Gaussian.append(
                            np.linalg.norm(
                                self.aSR[j, i, :] / integral_aSTF
                                - newSR[:, 1] / integral_Gaussian
                            )
                        )
        misfits_Yoffe = np.array(misfits_Yoffe)
        misfits_Gaussian = np.array(misfits_Gaussian)
        print(
            f"misfit Yoffe (10-50-90%): {np.percentile(misfits_Yoffe, 10):.2f} "
            f"{np.percentile(misfits_Yoffe, 50):.2f}"
            f"{np.percentile(misfits_Yoffe, 90):.2f}"
        )
        print(
            f"misfit Gaussian (10-50-90%): {np.percentile(misfits_Gaussian, 10):.2f} "
            f"{np.percentile(misfits_Gaussian, 50):.2f} "
            f"{np.percentile(misfits_Gaussian, 90):.2f}"
        )

        self.rise_time = interpolate_nan_from_neighbors(self.rise_time)
        self.tacc = interpolate_nan_from_neighbors(self.tacc)

        print(
            "slip rise_time (min, 50%, max)",
            np.amin(self.rise_time),
            np.median(self.rise_time),
            np.amax(self.rise_time),
        )
        print(
            "tacc (min, 50%, max)",
            np.amin(self.tacc),
            np.median(self.tacc),
            np.amax(self.tacc),
        )

    def upsample_fault(
        self,
        spatial_order,
        spatial_zoom,
        temporal_zoom,
        proj,
        use_Yoffe=False,
        time_smoothing_kernel_as_dt_fraction=0.5,
    ):
        "increase spatial and temporal resolution of kinematic model by interpolation"
        # time vector
        ndt2 = (self.ndt - 1) * temporal_zoom + 1
        ny2, nx2 = self.ny * spatial_zoom, self.nx * spatial_zoom
        # resampled source
        pf = FaultPlane()
        pf.init_spatial_arrays(nx2, ny2)
        pf.ndt = ndt2
        pf.init_aSR()

        pf.dt = self.dt / temporal_zoom
        pf.compute_time_array()

        # upsample spatially geometry (bilinear interpolation)
        allarr = np.array([self.x, self.y, self.depth])
        pf.x, pf.y, pf.depth = upsample_quantities(
            allarr, spatial_order=1, spatial_zoom=spatial_zoom, padding="extrapolate"
        )

        # upsample other quantities
        self.rake = np.unwrap(np.unwrap(self.rake, axis=0), axis=1)
        allarr = np.array([self.t0, self.strike, self.dip, self.rake])
        pf.t0, pf.strike, pf.dip, pf.rake = upsample_quantities(
            allarr, spatial_order, spatial_zoom, padding="edge"
        )
        # the interpolation may generate some acausality that we here prevent
        pf.t0 = np.maximum(pf.t0, np.amin(self.t0))

        allarr = np.array([self.slip1])
        (pf.slip1,) = upsample_quantities(
            allarr,
            spatial_order,
            spatial_zoom,
            padding="constant",
            minimize_block_average_variations=True,
        )
        pf.compute_latlon_from_xy(proj)
        pf.PSarea_cm2 = self.PSarea_cm2 / spatial_zoom**2
        ratio_potency = (
            np.sum(pf.slip1) * pf.PSarea_cm2 / (np.sum(self.slip1) * self.PSarea_cm2)
        )
        print(f"seismic potency ratio (upscaled over initial): {ratio_potency}")

        if use_Yoffe:
            allarr = np.array([self.rise_time, self.tacc])
            pf.rise_time, pf.tacc = upsample_quantities(
                allarr, spatial_order, spatial_zoom, padding="edge"
            )
            pf.rise_time = np.maximum(pf.rise_time, np.amin(self.rise_time))
            pf.tacc = np.maximum(pf.tacc, np.amin(self.tacc))
            # see comment above explaining why the 1.27 factor
            print("using ts = tacc / 1.27 to compute the regularized Yoffe")
            ts = pf.tacc / 1.27
            tr = pf.rise_time - 2.0 * ts
            tr = np.maximum(tr, ts)
            for j in range(pf.ny):
                for i in range(pf.nx):
                    for k, tk in enumerate(pf.myt):
                        pf.aSR[j, i, k] = pf.slip1[j, i] * regularizedYoffe(
                            tk, ts[j, i], tr[j, i]
                        )
        else:
            aSRa = np.zeros((pf.ny, pf.nx, self.ndt))
            for k in range(self.ndt):
                aSRa[:, :, k] = upsample_quantities(
                    np.array([self.aSR[:, :, k]]),
                    spatial_order,
                    spatial_zoom,
                    padding="constant",
                )

            # interpolate temporally the AST
            for j in range(pf.ny):
                for i in range(pf.nx):
                    # 1. upsample with linear interpolation
                    # 2. apply a gauss kernel to smooth out sharp edges
                    # 3. tapper the signal smoothly to 0 at both time ends
                    # 4. rescale SR to ensure integral (SR) = slip
                    f = interpolate.interp1d(self.myt, aSRa[j, i, :], kind="linear")
                    pf.aSR[j, i, :] = f(pf.myt)
                    tapper = cosine_taper(pf.ndt, self.dt / (pf.ndt * pf.dt))
                    pf.aSR[j, i, :] = tapper * ndimage.gaussian_filter1d(
                        pf.aSR[j, i, :],
                        time_smoothing_kernel_as_dt_fraction * self.dt / pf.dt,
                        mode="constant",
                    )
                    # With a cubic interpolation, the interpolated slip1 may be
                    # negative which does not make sense.
                    if pf.slip1[j, i] < 0:
                        pf.aSR[j, i, :] = 0
                        continue
                    # should be the SR
                    integral_STF = np.trapz(np.abs(pf.aSR[j, i, :]), dx=pf.dt)
                    if abs(integral_STF) > 0:
                        pf.aSR[j, i, :] = (
                            pf.slip1[j, i] * pf.aSR[j, i, :] / integral_STF
                        )
        return pf

    def compute_corrected_slip_for_differing_area(self, proj):
        """
        self.PSarea_cm2 may slightly differ from the patch area from the fault geometry
        (e.g. due to the projection)
        Therefore, we need to update slip to keep seismic potency (area*slip) unchanged
        """
        cm2m = 0.01
        km2m = 1e3
        PSarea_m2 = self.PSarea_cm2 * cm2m * cm2m
        self.compute_xy_from_latlon(proj)
        nx, ny = self.nx, self.ny
        # Compute actual dx and dy from coordinates
        dy = np.zeros((ny, nx))
        dx = np.zeros((ny, nx))
        # central difference for the inside
        coords = np.array((self.x, self.y, -km2m * self.depth))
        for i in range(0, nx):
            p0 = coords[:, 0 : ny - 2, i] - coords[:, 2:ny, i]
            dy[1 : ny - 1, i] = 0.5 * np.linalg.norm(p0, axis=0)
        # special case of 0 and ny-1
        p0 = coords[:, 1, :] - coords[:, 0, :]
        dy[0, :] = np.linalg.norm(p0, axis=0)
        p0 = coords[:, ny - 1, :] - coords[:, ny - 2, :]
        dy[ny - 1, :] = np.linalg.norm(p0, axis=0)
        # dx for coordinates
        for j in range(0, ny):
            p0 = coords[:, j, 0 : nx - 2] - coords[:, j, 2:nx]
            dx[j, 1 : nx - 1] = 0.5 * np.linalg.norm(p0, axis=0)
        p0 = coords[:, :, 1] - coords[:, :, 0]
        dx[:, 0] = np.linalg.norm(p0, axis=0)
        p0 = coords[:, :, nx - 1] - coords[:, :, nx - 2]
        dx[:, nx - 1] = np.linalg.norm(p0, axis=0)
        factor_area = dx[:, :] * dy[:, :] / PSarea_m2
        slip1 = self.slip1 * factor_area
        print(
            f"done correcting slip for area. \
The correcting factor ranges between {np.amin(factor_area)} and {np.amax(factor_area)}"
        )
        return slip1

    def compute_1d_dimension_arrays(self, spatial_zoom):
        self.spatial_zoom = spatial_zoom
        # Compute dimension arrays
        km2m = 1e3
        coords = np.array([self.x, self.y, -km2m * self.depth])
        ny, nx = coords.shape[1:3]

        center_row = np.diff(coords[:, (ny - 1) // 2, :])
        dx1 = np.linalg.norm(center_row, axis=0)
        center_col = np.diff(coords[:, :, (nx - 1) // 2])
        dy1 = np.linalg.norm(center_col, axis=0)

        # with this convention the first data point is in local coordinate (0,0)
        xb = np.insert(np.cumsum(dx1), 0, 0)
        yb = np.insert(np.cumsum(dy1), 0, 0)

        self.xb = np.pad(xb, ((1), (1)), "reflect", reflect_type="odd")
        self.yb = np.pad(yb, ((1), (1)), "reflect", reflect_type="odd")

        # we want to cover all the fault, that is up to -dx/2.
        # With this strategy We will cover a bit more than that,
        # but it is probably not a big deal

        ncrop = spatial_zoom - 1
        self.x_up = scipy.ndimage.zoom(
            self.xb, spatial_zoom, order=1, mode="grid-constant", grid_mode=True
        )[ncrop:-ncrop]
        self.y_up = scipy.ndimage.zoom(
            self.yb, spatial_zoom, order=1, mode="grid-constant", grid_mode=True
        )[ncrop:-ncrop]

        # used for the interpolation
        yg, xg = np.meshgrid(self.y_up, self.x_up)
        self.yx = np.array([yg.ravel(), xg.ravel()]).T

    def upsample_quantity_RGInterpolator_core(self, arr, method, is_slip=False):
        if is_slip:
            # tapper to 0 slip except at the top
            print("tapper slip to 0, except at the top (hardcoded)")
            padded_arr = np.pad(arr, ((1, 0), (1, 1)), "constant")
            padded_arr = np.pad(padded_arr, ((0, 1), (0, 0)), "edge")
        else:
            padded_arr = np.pad(arr, ((1, 1), (1, 1)), "edge")
        interp = RegularGridInterpolator([self.yb, self.xb], padded_arr)
        return (
            interp(self.yx, method=method)
            .reshape(self.x_up.shape[0], self.y_up.shape[0])
            .T
        )

    def upsample_quantity_RGInterpolator(self, arr, method, is_slip=False):
        my_array = self.upsample_quantity_RGInterpolator_core(arr, method, is_slip)
        minimize_block_average_variations = is_slip
        if minimize_block_average_variations:
            # inspired by Tinti et al. (2005) (Appendix A)
            # This is for the specific case of fault slip.
            # We want to preserve the seismic moment of each subfault after
            # interpolation
            # the rock rigidity is not know by this script
            # (would require some python binding of easi).
            # the subfault area is typically constant over the kinematic model
            # So we just want to perserve subfault average.
            print("trying to perserve subfault average...")
            my_array = np.maximum(0, my_array)
            best_misfit = float("inf")
            # The algorithm does not seem to converge, but produces better model
            # (given the misfit) that inital after 2-3 iterations
            niter = 3
            for i in range(niter):
                block_average = compute_block_mean(
                    my_array[1:-1, 1:-1], self.spatial_zoom
                )
                print(arr.shape, block_average.shape)
                correction = np.where(block_average != 0, arr / block_average, 0)
                # having a misfit as misfit = np.linalg.norm(correction) does not make
                # sense as for almost 0 slip, correction can be large
                misfit = np.linalg.norm(arr - block_average) / len(arr)
                if best_misfit > misfit:
                    if i == 0:
                        print(f"misfit at iter {i}: {misfit}")
                    else:
                        print(f"misfit improved at iter {i}: {misfit}")
                    best_misfit = misfit
                    best = np.copy(my_array)
                my_array = self.upsample_quantity_RGInterpolator_core(
                    correction * arr, method, is_slip
                )
                my_array = np.maximum(0, my_array)
            my_array = best
        return my_array

    def generate_netcdf_fl33(
        self, prefix, method, spatial_zoom, proj, write_paraview, slip_cutoff
    ):
        "generate netcdf files to be used with SeisSol friction law 33"

        if not os.path.exists("ASAGI_files"):
            os.makedirs("ASAGI_files")

        cm2m = 0.01
        # a kinematic model defines the fault quantities at the subfault center
        # a netcdf file defines the quantities at the nodes
        # therefore the extra_padding_layer=True, and the added di below
        cslip = self.compute_corrected_slip_for_differing_area(proj)
        print(f"applying a {slip_cutoff:.1f} cm cutoff to fault slip")
        self.compute_1d_dimension_arrays(spatial_zoom)

        upsampled_arrays = []

        slip = self.upsample_quantity_RGInterpolator(cslip, method, is_slip=True)
        slip[slip < slip_cutoff] = 0.0

        self.rake = np.deg2rad(self.rake)
        self.rake = np.unwrap(np.unwrap(self.rake, axis=0), axis=1)
        self.rake = np.rad2deg(self.rake)

        for arr in [self.t0, self.rake, self.rise_time, self.tacc]:
            upsampled_arrays.append(self.upsample_quantity_RGInterpolator(arr, method))

        rupttime, rake, rise_time, tacc = upsampled_arrays

        # upsampled duration, rise_time and acc_time may not be smaller than initial
        # values at least rise_time could lead to a non-causal kinematic model
        rupttime = np.maximum(rupttime, np.amin(self.t0))
        rise_time = np.maximum(rise_time, np.amin(self.rise_time))
        tacc = np.maximum(tacc, np.amin(self.tacc))

        rake_rad = np.radians(rake)
        strike_slip = slip * np.cos(rake_rad) * cm2m
        dip_slip = slip * np.sin(rake_rad) * cm2m

        def compute_rake_interp_low_slip(strike_slip, dip_slip, slip_threshold=0.1):
            "compute rake with, with interpolation is slip is too small"
            slip = np.sqrt(strike_slip**2 + dip_slip**2)
            slip_threshold = min(slip_threshold, 0.1 * np.amax(slip))
            rake = np.arctan2(dip_slip, strike_slip)
            rake[slip < slip_threshold] = np.nan
            nan_indices = np.isnan(rake)
            if nan_indices.any():
                # Create a meshgrid for interpolation
                x, y = np.meshgrid(np.arange(rake.shape[1]), np.arange(rake.shape[0]))

                # Flatten the arrays and remove NaNs
                x_flat = x[~nan_indices].flatten()
                y_flat = y[~nan_indices].flatten()
                rake_flat = rake[~nan_indices].flatten()
                # important for dealing with 2 pi rake jump
                rake_flat = np.unwrap(rake_flat)

                # Interpolate missing values using linear interpolation
                rake_interpolated_lin = griddata(
                    (x_flat, y_flat), rake_flat, (x, y), method="linear"
                )
                rake[~nan_indices] = rake_flat
                rake[nan_indices] = rake_interpolated_lin[nan_indices]
                nan_indices = np.isnan(rake)
                if nan_indices.any():
                    rake_interpolated_near = griddata(
                        (x_flat, y_flat), rake_flat, (x, y), method="nearest"
                    )
                    nan_indices = np.isnan(rake)
                    rake[nan_indices] = rake_interpolated_near[nan_indices]
            rake = gaussian_filter(rake, sigma=spatial_zoom / 2)
            return rake

        rake = compute_rake_interp_low_slip(strike_slip, dip_slip)

        ldataName = [
            "strike_slip",
            "dip_slip",
            "rupture_onset",
            "effective_rise_time",
            "acc_time",
            "rake_interp_low_slip",
        ]
        lgridded_myData = [strike_slip, dip_slip, rupttime, rise_time, tacc, rake]

        prefix2 = f"{prefix}_{spatial_zoom}_{method}"
        if write_paraview:
            # see comment above
            for i, sdata in enumerate(ldataName):
                writeNetcdf(
                    f"ASAGI_files/{prefix2}_{sdata}",
                    [self.x_up, self.y_up],
                    [sdata],
                    [lgridded_myData[i]],
                    paraview_readable=True,
                )
            writeNetcdf(
                f"ASAGI_files/{prefix2}_slip",
                [self.x_up, self.y_up],
                ["slip"],
                [slip * cm2m],
                paraview_readable=True,
            )
        writeNetcdf(
            f"ASAGI_files/{prefix2}", [self.x_up, self.y_up], ldataName, lgridded_myData
        )

    def compute_affine_vector_map(self):
        "compute the 2d vectors hh and hw and the offsets defining the"
        "2d affine map (parametric coordinates)."
        self.affine_map = {}
        cm2m = 0.01
        km2m = 1e3
        nx, ny = self.nx, self.ny
        p0 = np.array([self.x[0, 0], self.y[0, 0], -km2m * self.depth[0, 0]])
        p1 = np.array(
            [self.x[ny - 1, 0], self.y[ny - 1, 0], -km2m * self.depth[ny - 1, 0]]
        )
        p2 = np.array(
            [self.x[0, nx - 1], self.y[0, nx - 1], -km2m * self.depth[0, nx - 1]]
        )
        hw = p1 - p0
        dx2 = np.linalg.norm(hw) / (ny - 1)
        self.affine_map["hw"] = hw / np.linalg.norm(hw)
        hh = p2 - p0
        dx1 = np.linalg.norm(hh) / (nx - 1)
        self.affine_map["hh"] = hh / np.linalg.norm(hh)
        dx = np.sqrt(self.PSarea_cm2 * cm2m * cm2m)
        # a kinematic model defines the fault quantities at the subfault center
        # a netcdf file defines the quantities at the nodes
        # therefore the dx/2
        # the term dxi/np.sqrt(dx1*dx2) allows accounting for non-square patches
        self.affine_map["non_square_factor"] = dx / np.sqrt(dx1 * dx2)
        self.affine_map["t1"] = -np.dot(p0, self.affine_map["hh"])
        self.affine_map["t2"] = -np.dot(p0, self.affine_map["hw"])
        self.affine_map["dx1"] = dx1
        self.affine_map["dx2"] = dx2

    def write_ts_file(self, prefix):
        # Generate ts file containing mesh geometry
        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        vertex = np.zeros((4, 3))
        km2m = 1e3
        nx, ny = self.nx, self.ny

        p0 = np.array([self.x[0, 0], self.y[0, 0], -km2m * self.depth[0, 0]])
        p1 = np.array(
            [self.x[ny - 1, 0], self.y[ny - 1, 0], -km2m * self.depth[ny - 1, 0]]
        )
        p2 = np.array(
            [self.x[0, nx - 1], self.y[0, nx - 1], -km2m * self.depth[0, nx - 1]]
        )
        p3 = np.array(
            [
                self.x[ny - 1, nx - 1],
                self.y[ny - 1, nx - 1],
                -km2m * self.depth[ny - 1, nx - 1],
            ]
        )

        hh = self.affine_map["hh"]
        hw = self.affine_map["hw"]
        non_square_factor = self.affine_map["non_square_factor"]
        dx1 = self.affine_map["dx1"]
        dx2 = self.affine_map["dx2"]

        vertex[0, :] = p0 + 0.5 * (-hh * dx1 - hw * dx2) * non_square_factor
        vertex[1, :] = p2 + 0.5 * (hh * dx1 - hw * dx2) * non_square_factor
        vertex[2, :] = p3 + 0.5 * (hh * dx1 + hw * dx2) * non_square_factor
        vertex[3, :] = p1 + 0.5 * (-hh * dx1 + hw * dx2) * non_square_factor

        connect = np.zeros((2, 3), dtype=int)
        connect[0, :] = [1, 2, 3]
        connect[1, :] = [1, 3, 4]
        fname = f"tmp/{prefix}_fault.ts"
        with open(fname, "w") as fout:
            fout.write(
                "GOCAD TSURF 1\nHEADER {\nname:"
                + fname
                + "\nborder: true\n"
                + "mesh: false\n*border*bstone: true\n}\nTFACE\n"
            )
            for ivx in range(1, 5):
                fout.write(
                    "VRTX %s %s %s %s\n"
                    % (ivx, vertex[ivx - 1, 0], vertex[ivx - 1, 1], vertex[ivx - 1, 2])
                )

            for i in range(2):
                fout.write(
                    "TRGL %d %d %d\n" % (connect[i, 0], connect[i, 1], connect[i, 2])
                )
            fout.write("END\n")
        print(f"done writing {fname}")

    def trim(self):
        # some kinematic models are padded with zero to a large extent
        # this function removes the padding (but one)
        idy, idx = np.where(self.slip1 > 0)
        i0, i1 = max(0, min(idx) - 1), min(self.nx - 1, max(idx) + 1)
        j0, j1 = max(0, min(idy) - 1), min(self.ny - 1, max(idy) + 1)
        nx1 = i1 - i0 + 1
        ny1 = j1 - j0 + 1
        if (nx1 != self.nx) or (ny1 != self.ny):
            print("trimming the kinematic model")
            fp1 = FaultPlane()
            fp1.dx = self.dx
            fp1.dy = self.dy
            fp1.init_spatial_arrays(nx1, ny1)
            fp_attrs = [
                "lon",
                "lat",
                "depth",
                "slip1",
                "rake",
                "strike",
                "dip",
                "t0",
                "tacc",
                "rise_time",
            ]
            for attr in fp_attrs:
                setattr(fp1, attr, getattr(self, attr)[j0 : j1 + 1, i0 : i1 + 1])
            fp1.PSarea_cm2 = self.dx * self.dy * 1e10
            if self.ndt:
                fp1.dt = self.dt
                fp1.ndt = self.ndt
                fp1.myt = self.myt
                fp1.init_aSR()
                fp1.aSR[:, :, :] = self.aSR[j0 : j1 + 1, i0 : i1 + 1, :]
            return fp1
        else:
            return self

    def add_one_zero_slip_row_at_depth(self):
        fp1 = FaultPlane()
        fp1.dx = self.dx
        fp1.dy = self.dy
        fp1.init_spatial_arrays(self.nx, self.ny + 1)
        fp_attrs = [
            "lon",
            "lat",
            "depth",
            "slip1",
            "rake",
            "strike",
            "dip",
            "t0",
            "tacc",
            "rise_time",
        ]
        nx, ny = self.nx, self.ny
        for attr in fp_attrs:
            modified_data = np.zeros_like(fp1.lon)
            modified_data[:ny, :nx] = getattr(self, attr)
            modified_data[ny, :] = (
                2 * getattr(self, attr)[ny - 1, :] - getattr(self, attr)[ny - 2, :]
            )
            setattr(fp1, attr, modified_data)
        fp1.slip1[ny, :] = 0
        fp1.PSarea_cm2 = self.dx * self.dy * 1e10
        if self.ndt:
            fp1.dt = self.dt
            fp1.ndt = self.ndt
            fp1.myt = self.myt
            fp1.init_aSR()
            fp1.aSR[:ny, :nx, :] = self.aSR[:, :, :]
        return fp1
