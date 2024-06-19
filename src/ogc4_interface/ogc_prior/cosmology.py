import numpy as np
from pycbc.cosmology import get_cosmology, redshift_from_comoving_volume


def vc2z(vc, cosmology=None, interp=True):
    """Converts comoving volume to redshift.
    Parameters
    ----------
    vc : float or array_like
        Comoving volume.
    cosmology : pycbc.cosmology.Cosmology, optional
        Cosmology object. If None, the default cosmology is used.
    Returns
    -------
    float or array_like
        Redshift.
    """
    if cosmology is None:
        cosmology = get_cosmology()

    return redshift_from_comoving_volume(
        vc, cosmology=cosmology, interp=interp
    )


def z2vc(z, cosmology=None):
    """Converts redshift to comoving volume.
    Parameters
    ----------
    z : float or array_like
        Redshift.
    cosmology : pycbc.cosmology.Cosmology, optional
        Cosmology object. If None, the default cosmology is used.
    Returns
    -------
    float or array_like
        Comoving volume.
    """
    if cosmology is None:
        cosmology = get_cosmology()

    return cosmology.comoving_volume(z)


def dVcdz(z: float, cosmology=None):
    """Compute dVc/dz = (1+z)^2 D_{H} D_{A}^2 dΩ / E(z)
    Eq. 28, Hogg, 2000, https://arxiv.org/pdf/astro-ph/9905116
    """
    if cosmology is None:
        cosmology = get_cosmology()
    Dh = cosmology.hubble_distance
    Da = cosmology.angular_diameter_distance(z)
    Ez = cosmology.efunc(z)
    dOmega = 4 * np.pi
    return (1 + z) ** 2 * Dh * Da**2 * dOmega / Ez
