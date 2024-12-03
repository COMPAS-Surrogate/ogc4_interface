import os
from typing import List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .cacher import Cacher
from .event import Event
from .logger import logger
from .observing_run import ObservingRun
from .plotting import (
    CTOP,
    add_cntr,
    plot_event_mcz_uncertainty,
    plot_scatter,
    plot_weights,
)

URL = "https://github.com/COMPAS-Surrogate/ogc4_interface/raw/main/data/ogc4_mcz_weights.hdf5"


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
        self.weights: np.ndarray = weights

        ##
        self.n_events, self.n_z_bins, self.n_mc_bins = weights.shape

    @classmethod
    def load(
        cls,
        pastro_threshold=None,
        observing_runs=["O3a", "O3b"],
        filter_valid_mcz=True,
    ):
        fpath = Cacher(URL).fpath
        logger.info(f"Loading OGC4 McZ population from {fpath}")
        with h5py.File(fpath, "r") as f:
            mc_bins = f["mc_bins"][()]
            z_bins = f["z_bins"][()]
            event_data = pd.DataFrame.from_records(f["event_data"][()])
            event_data["Name"] = event_data["Name"].str.decode("utf-8")
            event_data["ObservingRun"] = event_data["Name"].apply(
                Event.name_to_observing_run
            )
            weights = f["weights"][()]

        assert all(
            [
                col in event_data.columns
                for col in [
                    "Name",
                    "srcmchirp",
                    "redshift",
                    "Pastro",
                    "ObservingRun",
                ]
            ]
        )
        assert weights.shape == (len(event_data), len(z_bins), len(mc_bins))
        res = cls(mc_bins, z_bins, event_data, weights)
        if pastro_threshold is not None:
            res = res.filter_events(
                pastro_threshold,
                observing_runs=observing_runs,
                filter_valid_mcz=filter_valid_mcz,
            )
        return res

    def __repr__(self):
        return "OGC4_McZ(n={}, bins=[{}, {}], {})".format(
            *self.weights.shape, self.runs_spanned
        )

    def plot_weights(self, title=False):
        weights = self.weights.copy()
        # compress the weights to 2D by summing over the 0th axis
        for i in range(len(weights)):  # normlise each event
            weights[i] = weights[i] / np.sum(weights[i])
        ax = plot_scatter(self.event_data[["redshift", "srcmchirp"]].values)
        ax = plot_weights(
            np.nansum(weights, axis=0), self.mc_bins, self.z_bins, ax=ax
        )
        Z, MC = np.meshgrid(self.z_bins, self.mc_bins)
        for i in range(len(weights)):
            add_cntr(ax, Z, MC, weights[i])
        fig = ax.get_figure()
        if title:
            fig.suptitle(
                f"OGC4 Population normalised weights (n={self.n_events})"
            )
        return ax

    def plot_individuals(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        weights = self.weights.copy()
        names = self.event_data["Name"].values
        Z, MC = np.meshgrid(self.z_bins, self.mc_bins)
        for i, name in tqdm(enumerate(names), total=len(names)):
            w = weights[i] / np.sum(weights[i])
            mc, z = self.event_data.loc[
                self.event_data["Name"] == name, ["srcmchirp", "redshift"]
            ].values[0]
            ax = plot_weights(w, self.mc_bins, self.z_bins)

            ax.set_title(f"{name} (mc={mc:.2f}M, z={z:.2f})")
            ax.scatter(z, mc, color=CTOP, s=1)
            add_cntr(ax, Z, MC, w)
            plt.savefig(f"{outdir}/weights_{name}.png")

    def get_pass_fail(self, threshold=0.95):
        mcz_pass = self.get_pass_mc_z()
        pastro_pass = self.get_pass_pastro(threshold)
        return [mcz and p for mcz, p in zip(mcz_pass, pastro_pass)]

    def get_pass_pastro(self, threshold=0.95):
        return [
            True if _pi >= threshold else False
            for _pi in self.event_data["Pastro"]
        ]

    def get_pass_mc_z(self):
        mc_rng = [self.mc_bins[0], self.mc_bins[-1]]
        z_rng = [self.z_bins[0], self.z_bins[-1]]
        mc_pass = [
            mc_rng[0] <= mc <= mc_rng[1] for mc in self.event_data["srcmchirp"]
        ]
        z_pass = [
            z_rng[0] <= z <= z_rng[1] for z in self.event_data["redshift"]
        ]
        return [mc and z for mc, z in zip(mc_pass, z_pass)]

    def filter_events(
        self,
        threshold=0.95,
        filter_valid_mcz=True,
        observing_runs=["O1", "O2", "O3a", "O3b"],
    ):
        pass_fail = self.get_pass_pastro(threshold)
        n_pass = sum(pass_fail)
        logger.info(
            f"Filtering events with Pastro > {threshold} [{self.n_events} -> {n_pass}]"
        )
        if filter_valid_mcz:
            pass_fail = [
                p and mcz for p, mcz in zip(pass_fail, self.get_pass_mc_z())
            ]
            logger.info(
                f"Filtering events with valid mcz [{n_pass} -> {sum(pass_fail)}]"
            )
        event_data = self.event_data[pass_fail]
        weights = self.weights[pass_fail]

        init_n = len(event_data)
        obs_mask = event_data["ObservingRun"].isin(observing_runs)
        weights = weights[obs_mask]
        event_data = event_data[obs_mask]
        if init_n != len(event_data):
            logger.info(
                f"Filtering events with ObservingRun == {observing_runs}: [{init_n} -> {len(event_data)}]"
            )
        return PopulationMcZ(self.mc_bins, self.z_bins, event_data, weights)

    def plot_event_mcz_estimates(self):
        fig, axes = plot_event_mcz_uncertainty(
            self.event_data, pass_fail=self.get_pass_fail()
        )
        axes[1].axvspan(0, self.mc_bins[0], color="k", alpha=0.1)
        axes[1].axvspan(self.mc_bins[-1], 100, color="k", alpha=0.1)
        return fig, axes

    @property
    def runs_spanned(self) -> List[str]:
        return list(set(self.event_data["ObservingRun"].unique()))

    @property
    def duration(self) -> float:
        return ObservingRun.get_total_durations(self.runs_spanned)
