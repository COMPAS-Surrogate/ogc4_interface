from ogc4_interface.population_mcz import PopulationMcZ
import numpy as np

def test_population(tmpdir):
    mc_bins = np.linspace(3, 40, 50)
    z_bins = np.linspace(0, 1, 100)
    p = PopulationMcZ(mc_bins, z_bins)
    assert p.n_events > 0
    assert p.weights.shape == (p.n_events, len(mc_bins) - 1, len(z_bins) - 1)
    ax = p.plot()
    fig = ax[0].get_figure()
    fig.savefig(f"{tmpdir}/population.png")
