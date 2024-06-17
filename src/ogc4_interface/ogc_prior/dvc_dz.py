import numpy as np
import os
from scipy.interpolate import CubicSpline
from .cosmology import redshift_to_comoving_volume, get_cosmology
from ..cacher import Cacher

import matplotlib.pyplot as plt


class dVcDz:
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
        vs = redshift_to_comoving_volume(zs, cosmology=self.cosmology)
        dVc_dz = np.gradient(vs , zs)
        np.savez(self.cache_fname, zs=zs, vs=vs, dVc_dz=dVc_dz)


    def _load_spline(self):
        data = np.load(self.cache_fname)
        self._spline = CubicSpline(data["zs"], data["dVc_dz"])


    def __call__(self, z):
        return self._spline(z)