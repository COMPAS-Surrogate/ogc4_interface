import numpy as np

from ogc4_interface.event import Event


def test_event(tmpdir):
    e = Event("GW150914_095045")
    mc_bins = np.linspace(3, 40, 50)
    z_bins = np.linspace(0, 1, 100)
    axes = e.plot_weights(mc_bins, z_bins)
    fig = axes[0].get_figure()
    fig.savefig(f"{tmpdir}/gw150914_event.png")

    e = Event("GW190521_030229")
    weights = e.get_weights(mc_bins, z_bins)
    assert np.sum(weights) == 0
    axes = e.plot_weights(mc_bins, z_bins)
    fig = axes[0].get_figure()
    fig.savefig(f"{tmpdir}/gw190521_event.png")
