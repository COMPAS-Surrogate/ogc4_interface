import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

Mc = 'srcmchirp'
Z = 'z'
CMAP = "Blues"

def plot_samples(samples, bounds, nbins=30, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    _fmt_ax(ax, bounds)
    mc, z = samples[:, 0], samples[:, 1]
    # numpy histogram2d
    H, xedges, yedges = np.histogram2d(z, mc, bins=nbins, range=[bounds[Z], bounds[Mc]])
    # pcolor plot
    cmp = ax.pcolor(
        xedges, yedges, H.T, cmap=CMAP,
        norm=LogNorm(vmin=1, vmax=H.max())
    )
    fig = ax.get_figure()
    cbar = fig.colorbar(cmp, ax=ax, orientation='horizontal')
    cbar.set_label(r"Counts")

    return ax

def _fmt_ax(ax, bounds=None):
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\mathcal{M}_{\rm{src}}$')
    if bounds:
        ax.set_xlim(bounds[Z])
        ax.set_ylim(bounds[Mc])

def plot_prob(prob_fn, bounds, grid_size=30, ax=None, logscale=False):
    if ax is None:
        fig, ax = plt.subplots()
    _fmt_ax(ax, bounds)

    mc_lin = np.linspace(*bounds[Mc], grid_size)
    z_lin = np.linspace(*bounds[Z], grid_size)
    mc_grid, z_grid = np.meshgrid(mc_lin, z_lin)

    prob = np.array([prob_fn(mc, z) for mc, z in zip(mc_grid.ravel(), z_grid.ravel())])
    prob = prob.reshape(mc_grid.shape)

    norm = None
    if logscale:
        norm = _get_norm(prob)

    cmp = ax.pcolor(
        z_grid, mc_grid, prob, cmap=CMAP,
        # edgecolors='white',
        # norm=LogNorm(vmin=0.001)
        norm=norm
    )

    # add colorbar above the axes
    fig = ax.get_figure()
    cbar = fig.colorbar(cmp, ax=ax, orientation='horizontal')
    cbar.set_label(r"$\pi(\mathcal{M}_{\rm{src}},z)$")
    return ax


def _get_norm(x):
    log_x = np.log(x)
    log_x = log_x[np.isfinite(log_x)]
    if len(log_x) == 0:
        return None
    return LogNorm(vmin=np.exp(log_x.min()), vmax=x.max())


def plot_weights(weights:np.ndarray, mc_bins, z_bins, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    _fmt_ax(ax, {Mc:[min(mc_bins), max(mc_bins)], Z:[min(z_bins), max(z_bins)]})
    cmp = ax.pcolor(
        z_bins, mc_bins, weights.T, cmap=CMAP,
        norm=_get_norm(weights)
    )
    # add colorbar above the axes
    fig = ax.get_figure()
    cbar = fig.colorbar(cmp, ax=ax, orientation='horizontal')
    cbar.set_label(r"$w_{z,\mathcal{M}}$")
    return ax

def plot_scatter(samples, bounds=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    _fmt_ax(ax, bounds)
    z, mc = samples[:, 0], samples[:, 1]
    ax.plot(z, mc, 'k.')
    return ax