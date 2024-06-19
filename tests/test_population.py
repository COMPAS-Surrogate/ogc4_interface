from ogc4_interface.population_mcz import PopulationMcZ
import numpy as np

def test_population(tmpdir):
    p = PopulationMcZ.load()
    assert p.n_events > 0
    ax = p.plot_weights()
    ax.get_figure().savefig(f"{tmpdir}/weights.png")
    # p.plot_individuals(f"{tmpdir}/individuals")
    fig, _ = p.plot_event_mcz_estimates()
    fig.savefig(f"{tmpdir}/event_mcz_estimates.png", bbox_inches='tight')

