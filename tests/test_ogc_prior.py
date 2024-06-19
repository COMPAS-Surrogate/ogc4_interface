import matplotlib.pyplot as plt

from ogc4_interface.ogc_prior import Prior


def test_prior(tmpdir, test_ini):
    prior = Prior(test_ini, detailed=True)
    assert prior.sample(1).shape == (1, 2)
    assert prior.log_prob(1, 1) != None

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
    prior.plot_samples(1000_000, ax=axes[0])
    prior.plot_prob(ax=axes[1], grid_size=30)
    fig.savefig(f"{tmpdir}/prior.png")
