import numpy as np

import numpy as np

from .summary import Summary
from .logger import logger
from .event import Event
from .cacher import Cacher
from .plotting import plot_weights

from tqdm.auto import tqdm
import os


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
            e = Event(name)
            weights[i, :, :] = e.get_weights(self.mc_bins, self.z_bins)
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
        return self._weights

    @property
    def n_events(self):
        return len(self.event_names)

    def __repr__(self):
        return f"OGC4_McZ(n={self.n_events}, bins=({len(self.mc_bins)}, {len(self.z_bins)}), pastro={self.pastro_threshold})"

    def plot(self):
        weights = self.weights.copy()
        # compress the weights to 2D by summing over the z axis
        weights = weights.sum(axis=2)
        ax = plot_weights(weights, self.mc_bins, self.z_bins)
        fig = ax.get_figure()
        fig.suptitle(self.label())
        return ax

