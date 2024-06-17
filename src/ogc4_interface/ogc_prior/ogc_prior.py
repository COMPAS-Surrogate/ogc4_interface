import numpy as np
from pycbc.distributions.utils import prior_from_config
from pycbc.inference import models, io
from pycbc.workflow import WorkflowConfigParser

from pycbc.distributions import JointDistribution
import matplotlib.pyplot as plt
import seaborn as sns
from .cosmology import redshift_from_comoving_volume, redshift_to_comoving_volume, get_cosmology
from .dvc_dz import dVcDz

from matplotlib.colors import LogNorm

class Prior:
    def __init__(self, ini, detailed=True, cosmology=None):
        self._prior = _read_prior_from_ini(ini)
        self.detailed = detailed
        if cosmology is None:
            cosmology = get_cosmology()
        self.cosmology = cosmology
        self._dvc_dz = dVcDz(cosmology=cosmology)

    def sample(self, n: int) -> np.ndarray:
        """
        Sample the prior distribution.

        Returns:
        np.ndarray: Array of samples of shape (n, 2) where
        the first column is the source chirp mass and the
        second column is the redshift.
        """
        samp = self._prior.rvs(n)
        z = redshift_from_comoving_volume(
            samp.comoving_volume,
            interp=self.detailed,
            cosmology=self.cosmology
        )
        return np.array([samp.srcmchirp, z]).T

    def log_prob(self, mc, z):
        vc = redshift_to_comoving_volume(z, cosmology=self.cosmology)
        logp_mcv = self._prior(srcmchirp=mc, comoving_volume=vc.value)
        log_dvdz = np.log(self._dvc_dz(z))
        logp_mcz = logp_mcv + log_dvdz
        return logp_mcz

    def prob(self, mc, z):
        return np.exp(self.log_prob(mc, z))

    @property
    def bounds(self):
        if not hasattr(self, '_bounds'):
            mc_bounds = [
                self._prior.bounds['srcmchirp'].min,
                self._prior.bounds['srcmchirp'].max
            ]
            vc_bounds = [
                self._prior.bounds['comoving_volume'].min,
                self._prior.bounds['comoving_volume'].max
            ]
            kwgs = dict(interp=self.detailed, cosmology=self.cosmology)
            z_bounds = [
                redshift_from_comoving_volume(vc_bounds[0], **kwgs),
                redshift_from_comoving_volume(vc_bounds[1], **kwgs)
            ]
            self._bounds = {
                'srcmchirp': mc_bounds,
                'z': z_bounds
            }
        return self._bounds

    def plot_samples(self, n, ax=None):
        samples = self.sample(n)
        if ax is None:
            fig, ax = plt.subplots()
        mc, z = samples[:, 0], samples[:, 1]
        sns.histplot(x=z, y=mc, ax=ax)
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$\mathcal{M}_{\rm{src}}$')
        ax.set_xlim(self.bounds['z'])
        ax.set_ylim(self.bounds['srcmchirp'])

        # add colorbar above the plot
        cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='horizontal')
        cbar.set_label('Counts')

        return ax

    def plot_prob(self, grid_size=30, ax=None, log_prob=False):
        if ax is None:
            fig, ax = plt.subplots()
        mc_lin = np.linspace(*self.bounds['srcmchirp'], grid_size)
        z_lin = np.linspace(*self.bounds['z'], grid_size)
        mc_grid, z_grid = np.meshgrid(mc_lin, z_lin)
        if log_prob:
            f, f_name = self.log_prob, 'log(prob)'
        else:
            f, f_name = self.prob, 'prob'

        prob = np.array([f(mc, z) for mc, z in zip(mc_grid.ravel(), z_grid.ravel())])
        prob = prob.reshape(mc_grid.shape)

        # imshow the log(prob) values (x = z, y = mc)
        cmp = ax.pcolor(
            z_grid, mc_grid, prob,
            edgecolors='white', cmap='Blues', linewidths=0.1
        )

        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$\mathcal{M}_{\rm{src}}$')
        ax.set_xlim(self.bounds['z'])
        ax.set_ylim(self.bounds['srcmchirp'])

        # add colorbar above the axes
        fig = ax.get_figure()
        cbar = fig.colorbar(cmp, ax=ax, orientation='horizontal')
        cbar.set_label(f_name)
        return ax


def _read_prior_from_ini(ini_fn: str) -> JointDistribution:
    config = WorkflowConfigParser(configFiles=[ini_fn])
    all_sections = config.sections()
    to_keep = ['prior-srcmchirp', 'prior-comoving_volume', 'waveform_transforms-redshift']
    to_remove = list(set(all_sections) - set(to_keep))
    for s in to_remove:
        config.remove_section(s)
    config.add_section('variable_params')
    config.set('variable_params', 'srcmchirp', '')
    config.set('variable_params', 'comoving_volume', '')

    prior = prior_from_config(config)
    return prior
