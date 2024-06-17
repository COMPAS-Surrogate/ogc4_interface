from pycbc.cosmology import redshift_from_comoving_volume, get_cosmology


def redshift_to_comoving_volume(z, cosmology=None):
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

