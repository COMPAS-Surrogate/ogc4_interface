import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Mc = 'srcmchirp'
Z = 'z'
CMAP = "Blues"

def plot_samples(samples, bounds, nbins=30, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    _fmt_ax(ax, bounds)
    mc, z = samples[:, 0], samples[:, 1]

    sns.histplot(x=z,y=mc, ax=ax, bins=nbins, cbar=True, cbar_kws=dict(
        label='Counts', cmap='Blues', orientation='horizontal'
    ))
    return ax

def _fmt_ax(ax, bounds):
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\mathcal{M}_{\rm{src}}$')
    ax.set_xlim(bounds[Z])
    ax.set_ylim(bounds[Mc])

def plot_prob(prob_fn, bounds, grid_size=30, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    _fmt_ax(ax, bounds)

    mc_lin = np.linspace(*bounds[Mc], grid_size)
    z_lin = np.linspace(*bounds[Z], grid_size)
    mc_grid, z_grid = np.meshgrid(mc_lin, z_lin)

    prob = np.array([prob_fn(mc, z) for mc, z in zip(mc_grid.ravel(), z_grid.ravel())])
    prob = prob.reshape(mc_grid.shape)

    cmp = ax.pcolor(
        z_grid, mc_grid, prob, cmap=CMAP,
        # edgecolors='white',
        # norm=LogNorm(vmin=0.001)
    )

    # add colorbar above the axes
    fig = ax.get_figure()
    cbar = fig.colorbar(cmp, ax=ax, orientation='horizontal')
    cbar.set_label(r"$\pi(\mathcal{M}_{\rm{src}},z)$")
    return ax


def plot_weights(weights:np.ndarray, mc_bins, z_bins, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    _fmt_ax(ax, {Mc:[min(mc_bins), max(mc_bins)], Z:[min(z_bins), max(z_bins)]})
    cmp = ax.pcolor(
        z_bins, mc_bins, weights.T, cmap=CMAP
    )
    # add colorbar above the axes
    fig = ax.get_figure()
    cbar = fig.colorbar(cmp, ax=ax, orientation='horizontal')
    cbar.set_label(r"$w_{z,\mathcal{M}}$")
    return ax
