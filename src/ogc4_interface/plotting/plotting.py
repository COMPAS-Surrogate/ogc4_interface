import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.colors import (to_rgba, ListedColormap)
from scipy.interpolate import interp1d
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, sRGBColor
from .cmap_generator import CMAP, CTOP

Mc = 'srcmchirp'
Z = 'z'



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
    vmin, vmax = np.exp(log_x.min()), x.max()
    # return LogNorm(vmin=np.exp(log_x.min()), vmax=x.max())
    return PowerNorm(gamma=0.3, vmin=vmin/10, vmax=vmax*3)


def plot_weights(weights:np.ndarray, mc_bins, z_bins, ax=None, contour=True):
    if ax is None:
        fig, ax = plt.subplots()
    _fmt_ax(ax, {Mc:[min(mc_bins), max(mc_bins)], Z:[min(z_bins), max(z_bins)]})
    cmp = ax.pcolor(
        z_bins, mc_bins, weights.T, cmap=CMAP,
        norm=_get_norm(weights)
    )

    if contour:
        Zb, MCb = np.meshgrid(z_bins, mc_bins)
        ax.contour(Zb, MCb, weights.T, levels=1, colors='tab:orange', linewidths=[0, 2], alpha=0.1)
    # add colorbar above the axes
    fig = ax.get_figure()
    cbar = fig.colorbar(cmp, ax=ax, orientation='horizontal')
    cbar.set_label(r"$w_{z,\mathcal{M}}$")
    return ax

def plot_scatter(samples, bounds=None, ax=None, color=CTOP):
    if ax is None:
        fig, ax = plt.subplots()
    _fmt_ax(ax, bounds)
    z, mc = samples[:, 0], samples[:, 1]
    ax.plot(z, mc, marker='.', c=color, lw=0)
    return ax