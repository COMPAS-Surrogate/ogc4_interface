import matplotlib.pyplot as plt
import numpy as np

import numpy as np
from tqdm.auto import tqdm

from .summary import Summary
from .logger import logger
from .event import Event
from .cacher import Cacher
from .plotting import plot_weights, CTOP
from tqdm.auto import tqdm
import os

from scipy.ndimage import gaussian_filter


class PopulationMcZ:
    def __init__(self, mc_bins: np.array, z_bins: np.array, pastro_threshold=0.95):
        self.pastro_threshold = pastro_threshold
        self.mc_bins = mc_bins
        self.z_bins = z_bins

    @property
    def event_names(self):
        if not hasattr(self, "_event_names"):
            s = Summary.load()
            self._event_names = s.get_pastro_threholded_event_names(self.pastro_threshold)
        return self._event_names

    def _build_weights_matrix(self):
        weights = np.zeros(
            (len(self.event_names),  len(self.z_bins), len(self.mc_bins) )
        )
        for i, name in enumerate(tqdm(self.event_names, desc="Building weights matrix")):
            try:
                e = Event(name)
                weights[i, :, :] = e.get_weights(self.mc_bins, self.z_bins)
            except Exception as e:
                logger.warning(f"Failed to get weights for {name}: {e}")
        np.save(self.weights_fname, weights)

    @property
    def weights_fname(self):
        return f"{Cacher.cache_dir}/ogc4_weights_{self.label()}.npy"

    def label(self):
        mc_label = f"Mc_{len(self.mc_bins)}_{self.mc_bins[0]}_{self.mc_bins[-1]}"
        z_label = f"z_{len(self.z_bins)}_{self.z_bins[0]}_{self.z_bins[-1]}"
        return f"{mc_label}_{z_label}_pastro_{self.pastro_threshold}"

    @property
    def weights(self):
        if not hasattr(self, "_weights"):
            if not os.path.exists(self.weights_fname):
                self._build_weights_matrix()
            self._weights = np.load(self.weights_fname)

        # drop any slice with weights that sum to < 1e-5 (THESE ARE EMPTY SLICES)
        # weights_sum = np.sum(self._weights, axis=(1, 2))
        # mask = weights_sum > 1e-5
        # self._weights = self._weights[mask]

        return self._weights

    @property
    def n_events(self):
        n_events, _, _ = self.weights.shape
        return n_events

    def __repr__(self):
        return f"OGC4_McZ(n={self.n_events}, bins=({len(self.mc_bins)}, {len(self.z_bins)}), pastro={self.pastro_threshold})"

    def plot(self):
        weights = self.weights.copy()
        # compress the weights to 2D by summing over the 0th axis
        for i in range(len(weights)): # normlise each event
            weights[i] = weights[i] / np.sum(weights[i])


        s = Summary.load()
        ax = s.plot(pastro_threshold=self.pastro_threshold, color=CTOP)
        ax = plot_weights(np.nansum(weights, axis=0), self.mc_bins, self.z_bins,ax=ax)

        Z, MC = np.meshgrid(self.z_bins, self.mc_bins)
        for i in range(len(weights)):
            ax.contour(Z, MC, gaussian_filter(weights[i], 2).T, levels=1, colors=CTOP, linewidths=[0,2], alpha=0.1)


        fig = ax.get_figure()
        fig.suptitle(f"OGC4 Population normalised weights (n={self.n_events})")
        return ax


    def plot_individuals(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        weights = self.weights.copy()
        s = Summary.load()
        for i, name in tqdm(enumerate(self.event_names)):
            w = weights[i]
            w = w / np.sum(w)
            # get mc and z values for this event from summary
            mc, z = s.get_mcz_for(name)
            ax = plot_weights(w, self.mc_bins, self.z_bins)
            ax.set_title(f"{name} (mc={mc:.2f}M, z={z:.2f})")
            ax.scatter(z, mc, color='k', s=1)
            plt.savefig(f"{outdir}/weights_{name}.png")


