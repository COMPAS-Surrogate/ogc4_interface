import pandas as pd

from ogc4_interface.plotting.plotting import plot_event_mcz_uncertainty
from ogc4_interface.summary import Summary


def test_plot_event_mcz_uncertainty(tmpdir):
    s = Summary.load()
    d = s._data[
        [
            "Name",
            "srcmchirp",
            "redshift",
            "Pastro",
            "srcmchirp_plus",
            "srcmchirp_minus",
            "redshift_plus",
            "redshift_minus",
        ]
    ]
    # sample 10
    # d = d.sample(10)
    fig, _ = plot_event_mcz_uncertainty(d)
    fig.savefig(f"{tmpdir}/event_mcz_uncertainty.png", bbox_inches="tight")
