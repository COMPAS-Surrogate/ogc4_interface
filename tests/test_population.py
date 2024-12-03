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

    p = p.filter_events(
        threshold=0, filter_valid_mcz=False  # observing_runs=["O3a", "O3b"],
    )
    fig, _ = p.plot_event_mcz_estimates()
    fig.savefig(
        f"{tmpdir}/event_mcz_estimates_only_o3.png",
        bbox_inches="tight",
        dpi=300,
    )

    p = p.filter_events(
        threshold=0, observing_runs=["O3a", "O3b"], filter_valid_mcz=False
    )
    fig, _ = p.plot_event_mcz_estimates()
    fig.savefig(
        f"{tmpdir}/event_mcz_estimates_pastro_check.png",
        bbox_inches="tight",
        dpi=300,
    )

    p = p.filter_events(
        threshold=0.95, observing_runs=["O3a", "O3b"], filter_valid_mcz=True
    )
    fig, _ = p.plot_event_mcz_estimates()
    fig.savefig(
        f"{tmpdir}/event_mcz_estimates_pastro_check_and_mcz_check.png",
        bbox_inches="tight",
        dpi=300,
    )

    ax = p.plot_weights(title=True)
    ax.get_figure().savefig(f"{tmpdir}/weights.png", dpi=300)

    assert p.duration > 0.1  # years
