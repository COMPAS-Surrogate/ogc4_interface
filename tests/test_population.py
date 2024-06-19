import numpy as np

from ogc4_interface.population_mcz import PopulationMcZ


def test_population(tmpdir):
    p = PopulationMcZ.load()
    assert p.n_events > 0
    fig, _ = p.plot_event_mcz_estimates()
    fig.savefig(
        f"{tmpdir}/event_mcz_estimates.png", bbox_inches="tight", dpi=300
    )

    p = p.filter_events(threshold=0.95)
    ax = p.plot_weights()
    ax.get_figure().savefig(f"{tmpdir}/weights.png", dpi=300)
