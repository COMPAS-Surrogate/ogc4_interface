import numpy as np
import matplotlib.pyplot as plt

from .utils import BASE_URL
from .cacher import Cacher
from requests import HTTPError
from .logger import logger
from .ogc_prior import Prior
from .plotting import plot_samples, plot_weights

import h5py

POSTERIOR_URL = BASE_URL + "/posterior/{}-PYCBC-POSTERIOR-IMRPhenomXPHM.hdf"
INI_URL = BASE_URL + "/inference_configuration/inference-{}.ini"


class Event:
    def __init__(self, name: str):
        self.name = name

    @property
    def prior(self):
        if not hasattr(self, "_prior"):
            self._prior = Prior(self.ini_fn)
        return self._prior

    @property
    def posterior_samples(self):
        if not hasattr(self, "_posterior_samples"):
            self._posterior_samples = self._load_mcz_from_hdf()
        return self._posterior_samples

    def _load_mcz_from_hdf(self) -> np.ndarray:
        """Returns [[mchirp, z], ...] Shape: (n_samples, 2) from the posterior file"""
        with h5py.File(self.posterior_fn, 'r') as fp:
            samples = fp['samples']
            z = samples['redshift'][()]
            mchirp = samples['srcmchirp'][()]
            return np.array([mchirp, z]).T

    def download_data(self):
        try:
            logger.debug(f"Init {self.posterior_fn}")
            logger.debug(f"Init {self.ini_fn}")

        except HTTPError:
            logger.error(f"Skipping download for {self.name}... Cant find files to download!")

    @property
    def posterior_fn(self) -> str:
        return Cacher.get(POSTERIOR_URL.format(self.name))

    @property
    def ini_fn(self) -> str:
        return Cacher.get(INI_URL.format(self.name))


    def plot(self, axes=None, nbins=30):
        if axes is None:
            fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

        self.prior.plot_prob(ax=axes[0], grid_size=nbins)
        plot_samples(self.posterior_samples, bounds=self.prior.bounds, ax=axes[1], nbins=nbins)
        axes[0].set_title("Prior")
        axes[1].set_title("Posterior")
        fig = axes[0].get_figure()
        fig.suptitle(self.name)
        return axes

    def plot_weights(self, mc_bins, z_bins, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 3,  figsize=(15, 6))
        self.plot(axes[:2])
        weights = self.get_weights(mc_bins, z_bins)
        plot_weights(weights, mc_bins, z_bins, ax=axes[2])
        axes[2].set_title("Weights")
        return axes


    def get_weights(self, mc_bins:np.array, z_bins:np.array)->np.ndarray:
        """
        Returns the weights for the mcz_grid for the event.

        Args:
            mc_bins (np.array): The chirp mass bins.
            z_bins (np.array): The redshift bins.

        Returns:
            np.ndarray: The weights for the mcz_grid (n_z_bins, n_mc_bins)
        """
        n_z_bins, n_mc_bins = len(z_bins), len(mc_bins)
        weights = np.zeros((n_z_bins, n_mc_bins))

        for mc, z in self.posterior_samples:
            mc_bin = np.argmin(np.abs(mc_bins - mc))
            z_bin = np.argmin(np.abs(z_bins - z))
            if mc_bin < n_mc_bins and z_bin < n_z_bins:
                weights[z_bin, mc_bin] += 1 / self.prior.prob(mc=mc, z=z)

        weights /= len(self.posterior_samples)

        return weights
