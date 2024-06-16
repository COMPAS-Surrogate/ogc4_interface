import numpy as np
import pandas as pd

from .summary import Summary
from .utils import BASE_URL
from .cacher import Cacher
from requests import HTTPError
from .logger import logger

import h5py


POSTERIOR_URL = BASE_URL + "/posterior/{}-PYCBC-POSTERIOR-IMRPhenomXPHM.hdf"
INI_URL = BASE_URL + "/inference_configuration/inference-{}.ini"


class Event:
    def __init__(self, name: str):
        self.name = name

    def prior(self, ):

    def _get_mcz_prior_from_ini(self):
        config = WorkflowConfigParser(configFiles=[self.ini_fn])
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

    @property
    def posterior(self):
        if not hasattr(self, "_posterior"):
            self._posterior = self._load_mcz_from_hdf()
        return self._posterior

    def _load_mcz_from_hdf(self) -> np.ndarray:
        with h5py.File(self.posterior_fn, 'r') as fp:
            samples = fp['samples']
            z = samples['redshift'][()]
            mchirp = samples['srcmchirp'][()]

            return np.ndarray([mchirp, z])

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


def read_priors(ini: str, n_samples: int = 5000):

    samples = prior.rvs(n_samples)
    z = redshift_from_comoving_volume(samples.comoving_volume)
    return pd.DataFrame(dict(
        srcmchirp=samples.srcmchirp,
        redshift=z,
        comoving_volume=samples.comoving_volume
    ))


