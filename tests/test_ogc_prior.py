import matplotlib.pyplot as plt
from ogc4_interface.ogc_prior import Prior
import seaborn as sns
import numpy as np


def test_prior(tmpdir, test_ini):
    prior = Prior(test_ini, detailed=False)
    assert prior.sample(1).shape == (1, 2)
    assert prior.log_prob(1, 1) != None


    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
    prior.plot_samples(10000, ax=axes[0])
    prior.plot_prob(ax=axes[1])
    fig.savefig(f"{tmpdir}/prior.png")
