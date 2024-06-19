import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from ogc4_interface.cacher import Cacher
from ogc4_interface.ogc_prior.cosmology import dVcdz, get_cosmology, z2vc


def test_dvdz(tmpdir):
    dvc_dz = dVcDzNumerical(n_grid=1000)
    zs = np.linspace(0, 10, 10000)
    pred_dvdz = dvc_dz(zs)
    analytical_dvdz = dVcdz(zs, cosmology=dvc_dz.cosmology)

    # plot error histogram
    error = np.abs(analytical_dvdz.value - pred_dvdz)
    fig, ax = plt.subplots(2, 1)
    ax[0].hist(error, bins=np.geomspace(1e-5, 1e-4, 100), histtype="step")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Error")

    cosmology = dvc_dz.cosmology
    inv_d3 = cosmology.hubble_distance**-3
    ax[1].plot(zs, analytical_dvdz * inv_d3)
    ax[1].plot(zs, pred_dvdz * inv_d3)
    ax[1].set_xlabel("z")
    ax[1].set_ylabel("dVc/dz")
    plt.savefig(f"{tmpdir}/dVcDz_error.png")


class dVcDzNumerical:
    def __init__(self, cosmology=None, n_grid=1000):
        if cosmology is None:
            cosmology = get_cosmology()
        self.cosmology = cosmology
        self._spline = None
        self._ngrid = n_grid

        if not self.exists:
            self._build_cache()
        self._load_spline()

    @property
    def cache_fname(self):
        return f"{Cacher.cache_dir}/dVcDz_{self.cosmology.name}_n{self._ngrid}.npz"

    @property
    def exists(self):
        return os.path.exists(self.cache_fname)

    def _build_cache(self):
        zs = np.linspace(0, 10, 1000)
        vs = z2vc(zs, cosmology=self.cosmology)
        dVc_dz = np.gradient(vs, zs)
        np.savez(self.cache_fname, zs=zs, vs=vs, dVc_dz=dVc_dz)

    def _load_spline(self):
        data = np.load(self.cache_fname)
        self._spline = CubicSpline(data["zs"], data["dVc_dz"])

    def __call__(self, z):
        return self._spline(z)
