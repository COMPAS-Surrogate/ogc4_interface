import h5py
import numpy as np
from tqdm.auto import tqdm

from .cacher import Cacher
from .event import Event
from .logger import logger
from .summary import Summary


def build_cache(mc_bins=None, z_bins=None, fname=None):
    s = Summary.load()
    if mc_bins is None:
        mc_bins = np.linspace(3, 40, 50)
    if z_bins is None:
        z_bins = np.linspace(0, 1, 100)
    if fname is None:
        fname = f"{Cacher.cache_dir}/population.hdf5"

    weights = _get_weights(s, mc_bins, z_bins)
    with h5py.File(fname, "w") as f:
        f.create_dataset("mc_bins", data=mc_bins)
        f.create_dataset("z_bins", data=z_bins)
        f.create_dataset("weights", data=weights)
        f.create_dataset("event_data", data=s.data_records)


def _get_weights(s: Summary, mc_bins, z_bins) -> np.ndarray:
    weights = np.zeros((len(s), len(z_bins), len(mc_bins)))
    for i, name in enumerate(
        tqdm(s.event_names, desc="Building weights matrix")
    ):
        try:
            e = Event(name)
            weights[i, :, :] = e.get_weights(mc_bins=mc_bins, z_bins=z_bins)
        except Exception as e:
            logger.warning(f"Failed to get weights for {name}: {e}")
            weights[i, :, :] = np.nan
    return weights
