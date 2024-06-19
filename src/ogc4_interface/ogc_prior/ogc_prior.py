import numpy as np
from pycbc.distributions import JointDistribution
from pycbc.distributions.utils import prior_from_config
from pycbc.workflow import WorkflowConfigParser

from ..plotting import plot_prob, plot_samples
from .cosmology import dVcdz, get_cosmology, vc2z, z2vc

Mc = "srcmchirp"
Z = "z"
VC = "comoving_volume"


class Prior:
    def __init__(self, ini, detailed=True, cosmology=None):
        self._prior = _read_prior_from_ini(ini)
        self.detailed = detailed
        if cosmology is None:
            cosmology = get_cosmology()
        self.cosmology = cosmology

    def sample(self, n: int) -> np.ndarray:
        """
        Sample the prior distribution.

        Returns:
        np.ndarray: Array of samples of shape (n, 2) where
        the first column is the source chirp mass and the
        second column is the redshift.
        """
        samp = self._prior.rvs(n)
        z = vc2z(
            samp.comoving_volume,
            interp=self.detailed,
            cosmology=self.cosmology,
        )
        return np.array([samp.srcmchirp, z]).T

    def log_prob(self, mc, z):
        vc = z2vc(z, cosmology=self.cosmology)
        _dvcdz = dVcdz(z, cosmology=self.cosmology)
        # assert same units
        assert vc.unit == _dvcdz.unit
        logp_mcv = self._prior(srcmchirp=mc, comoving_volume=vc.value)
        log_dvdz = np.log(_dvcdz.value)
        logp_mcz = logp_mcv + log_dvdz
        return logp_mcz

    def prob(self, mc, z):
        return np.exp(self.log_prob(mc, z))

    @property
    def bounds(self):
        if not hasattr(self, "_bounds"):
            kwgs = dict(interp=self.detailed, cosmology=self.cosmology)
            self._bounds = {
                Mc: [self._prior.bounds[Mc].min, self._prior.bounds[Mc].max],
                Z: [
                    vc2z(self._prior.bounds[VC].min, **kwgs),
                    vc2z(self._prior.bounds[VC].max, **kwgs),
                ],
            }
        return self._bounds

    def plot_samples(self, n, ax=None):
        samples = self.sample(n)
        return plot_samples(samples, self.bounds, ax=ax)

    def plot_prob(self, ax=None, grid_size=30):
        return plot_prob(self.prob, self.bounds, grid_size=grid_size, ax=ax)


def _read_prior_from_ini(ini_fn: str) -> JointDistribution:
    config = WorkflowConfigParser(configFiles=[ini_fn])
    all_sections = config.sections()
    to_keep = [
        "prior-srcmchirp",
        "prior-comoving_volume",
        "waveform_transforms-redshift",
    ]
    to_remove = list(set(all_sections) - set(to_keep))
    for s in to_remove:
        config.remove_section(s)
    config.add_section("variable_params")
    config.set("variable_params", Mc, "")
    config.set("variable_params", "comoving_volume", "")

    prior = prior_from_config(config)
    return prior
