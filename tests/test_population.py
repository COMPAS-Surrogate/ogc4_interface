from ogc4_interface.population_mcz import PopulationMcZ


def test_population(tmpdir):
    p = PopulationMcZ.load()
    assert p.n_events > 0
    fig, _ = p.plot_event_mcz_estimates()
    fig.savefig(
        f"{tmpdir}/event_mcz_estimates.png", bbox_inches="tight", dpi=300
    )


    ax = p.plot_weights(title=True)
    ax.get_figure().savefig(f"{tmpdir}/weights_orig.png", dpi=300)

    p = p.filter_events(threshold=0.95, observing_runs=["O3a", "O3b"])
    ax = p.plot_weights(title=True)
    ax.get_figure().savefig(f"{tmpdir}/weights.png", dpi=300)
