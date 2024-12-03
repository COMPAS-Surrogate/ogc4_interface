from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
from requests import HTTPError
from tqdm.auto import tqdm

from .cacher import BASE_URL, Cacher
from .logger import logger
from .observing_run import ObservingRun
from .ogc_prior import Prior
from .plotting import plot_samples, plot_weights

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
        with h5py.File(self.posterior_fn, "r") as fp:
            samples = fp["samples"]
            z = samples["redshift"][()]
            mchirp = samples["srcmchirp"][()]
            return np.array([mchirp, z]).T

    def download_data(self):
        try:
            logger.debug(f"Init {self.posterior_fn}")
            logger.debug(f"Init {self.ini_fn}")

        except HTTPError:
            logger.error(
                f"Skipping download for {self.name}... Cant find files to download!"
            )

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
        plot_samples(
            self.posterior_samples,
            bounds=self.prior.bounds,
            ax=axes[1],
            nbins=nbins,
        )
        axes[0].set_title("Prior")
        axes[1].set_title("Posterior")
        fig = axes[0].get_figure()
        fig.suptitle(self.name)
        return axes

    def plot_weights(self, mc_bins, z_bins, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        self.plot(axes[:2])
        weights = self.get_weights(mc_bins, z_bins)
        plot_weights(weights, mc_bins, z_bins, ax=axes[2])
        axes[2].set_title("Weights")
        return axes

    def get_weights(self, mc_bins: np.array, z_bins: np.array) -> np.ndarray:
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

        for mc, z in tqdm(
            self.posterior_samples, desc=f"Weights[{self.name}]"
        ):
            # check if the mc and z are within the bins
            in_mbin = mc_bins[0] <= mc <= mc_bins[-1]
            in_zbin = z_bins[0] <= z <= z_bins[-1]
            if not (in_mbin and in_zbin):
                continue

            mc_bin = np.argmin(np.abs(mc_bins - mc))
            z_bin = np.argmin(np.abs(z_bins - z))
            weights[z_bin, mc_bin] += 1 / self.prior.prob(mc=mc, z=z)

        weights /= len(self.posterior_samples)

        return weights

    @property
    def trigger_date(self) -> datetime:
        return self.name_to_date(self.name)

    @property
    def observing_run(self) -> str:
        return self.name_to_observing_run(self.name)

    @staticmethod
    def name_to_date(name: str) -> datetime:
        return datetime.strptime(name, "GW%y%m%d_%H%M%S")

    @staticmethod
    def name_to_observing_run(name: str) -> str:
        return ObservingRun.from_date(Event.name_to_date(name)).name
