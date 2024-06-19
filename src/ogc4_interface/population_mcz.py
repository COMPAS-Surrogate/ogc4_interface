import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .summary import Summary
from .logger import logger
from .event import Event
from .cacher import Cacher
from .plotting import plot_weights, CTOP, plot_scatter, add_cntr, plot_event_mcz_uncertainty
from tqdm.auto import tqdm
import os

import h5py





class PopulationMcZ:
    def __init__(
            self,
            mc_bins: np.array,
            z_bins: np.array,
            event_data: pd.DataFrame,
            weights: np.ndarray,
    ):
        self.mc_bins = mc_bins
        self.z_bins = z_bins
        self.event_data = event_data
        self.weights = weights

        ##
        self.n_events, self.n_z_bins, self.n_mc_bins = weights.shape


    @classmethod
    def load(cls, fname=None):

        if fname is None:
            fname = f"{Cacher.cache_dir}/population.hdf5"

        with h5py.File(fname, 'r') as f:
            mc_bins = f['mc_bins'][()]
            z_bins = f['z_bins'][()]
            event_data = pd.DataFrame.from_records(f['event_data'][()])
            event_data['Name'] = event_data['Name'].astype(str)
            weights = f['weights'][()]
        assert all([col in event_data.columns for col in ['Name', 'srcmchirp', 'redshift', 'Pastro']])
        assert weights.shape == (len(event_data), len(z_bins), len(mc_bins))
        return cls(mc_bins, z_bins, event_data, weights)

    def __repr__(self):
        return "OGC4_McZ(n={}, bins=[{}, {}]".format(*self.weights.shape)

    def plot_weights(self):
        weights = self.weights.copy()
        # compress the weights to 2D by summing over the 0th axis
        for i in range(len(weights)): # normlise each event
            weights[i] = weights[i] / np.sum(weights[i])
        ax = plot_scatter(self.event_data[['redshift', 'srcmchirp']].values)
        ax = plot_weights(np.nansum(weights, axis=0), self.mc_bins, self.z_bins,ax=ax)
        Z, MC = np.meshgrid(self.z_bins, self.mc_bins)
        for i in range(len(weights)):
            add_cntr(ax, Z, MC, weights[i])
        fig = ax.get_figure()
        fig.suptitle(f"OGC4 Population normalised weights (n={self.n_events})")
        return ax


    def plot_individuals(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        weights = self.weights.copy()
        names = self.event_data['Name'].values
        Z, MC = np.meshgrid(self.z_bins, self.mc_bins)
        for i, name in tqdm(enumerate(names), total=len(names)):
            w = weights[i] / np.sum(weights[i])
            mc, z = self.event_data.loc[self.event_data['Name'] == name, ['srcmchirp', 'redshift']].values[0]
            ax = plot_weights(w, self.mc_bins, self.z_bins)

            ax.set_title(f"{name} (mc={mc:.2f}M, z={z:.2f})")
            ax.scatter(z, mc, color=CTOP, s=1)
            add_cntr(ax, Z, MC, w)
            plt.savefig(f"{outdir}/weights_{name}.png")


    def get_pass_fail(self, threshold=0.95):
        mc_rng = [self.mc_bins[0], self.mc_bins[-1]]
        z_rng = [self.z_bins[0], self.z_bins[-1]]
        mc_pass = [mc_rng[0] <= mc <= mc_rng[1] for mc in self.event_data['srcmchirp']]
        z_pass = [z_rng[0] <= z <= z_rng[1] for z in self.event_data['redshift']]
        pastro_pass = [True if _pi >= threshold else False for _pi in self.event_data['Pastro']]
        return [mc and z and p for mc, z, p in zip(mc_pass, z_pass, pastro_pass)]

    def plot_event_mcz_estimates(self):
        fig, axes = plot_event_mcz_uncertainty(self.event_data, pass_fail=self.get_pass_fail())
        axes[1].axvspan(0, self.mc_bins[0], color='k', alpha=0.1)
        axes[1].axvspan(self.mc_bins[-1], 100, color='k', alpha=0.1)
        return fig, axes





