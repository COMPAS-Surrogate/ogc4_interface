from ogc4_interface.population_mcz import PopulationMcZ
import numpy as np

def test_population(tmpdir):
    mc_bins = np.linspace(3, 40, 50)
    z_bins = np.linspace(0, 1, 100)
    p = PopulationMcZ(mc_bins, z_bins)
    assert p.n_events > 0
    assert p.weights.shape[1:] == (len(z_bins), len(mc_bins))
    ax = p.plot()
    fig = ax.get_figure()
    fig.savefig(f"{tmpdir}/population.png")
    # p.plot_individuals(f'{tmpdir}/individuals')