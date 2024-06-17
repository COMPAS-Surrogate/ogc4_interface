import matplotlib.pyplot as plt
from ogc4_interface.ogc_prior import Prior
from ogc4_interface.ogc_prior.dvc_dz import dVcDz
import seaborn as sns
import numpy as np
from scipy.interpolate import CubicSpline


def test_prior(tmpdir, test_ini):
    prior = Prior(test_ini, detailed=True)
    # assert prior.sample(1).shape == (1, 2)
    # assert prior.log_prob(1, 1) != None

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
    prior.plot_samples(1000_000, ax=axes[0])
    prior.plot_prob(ax=axes[1], grid_size=50)
    fig.savefig(f"{tmpdir}/prior.png")

def test_dvdz(tmpdir):
    dvc_dz = dVcDz(n_grid=1000)
    dvc_dz_2 = dVcDz(n_grid=10_000)
    data = np.load(dvc_dz_2.cache_fname)
    zs = data["zs"]
    expected_dvdz = data["dVc_dz"]
    pred_dvdz = dvc_dz(zs)
    # plot error histogram
    error = np.abs(expected_dvdz - pred_dvdz)


    fig, ax = plt.subplots(2,1)
    ax[0].hist(error, bins=np.geomspace(1e-5, 1e-4,  100), histtype="step")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Error")

    cosmology = dvc_dz.cosmology
    inv_d3 = cosmology.hubble_distance ** -3
    ax[1].plot(zs, expected_dvdz*inv_d3)
    ax[1].plot(zs, pred_dvdz*inv_d3)
    ax[1].set_xlabel("z")
    ax[1].set_ylabel("dVc/dz")
    plt.savefig(f"{tmpdir}/dVcDz_error.png")

