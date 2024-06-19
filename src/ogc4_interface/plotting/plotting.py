import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, sRGBColor
from matplotlib.colors import ListedColormap, LogNorm, PowerNorm, to_rgba
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

from .cmap_generator import CMAP, CTOP, get_colors_from_cmap

Mc = "srcmchirp"
Z = "z"


def plot_samples(samples, bounds, nbins=30, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    _fmt_ax(ax, bounds)
    mc, z = samples[:, 0], samples[:, 1]
    # numpy histogram2d
    H, xedges, yedges = np.histogram2d(
        z, mc, bins=nbins, range=[bounds[Z], bounds[Mc]]
    )
    # pcolor plot
    cmp = ax.pcolor(
        xedges, yedges, H.T, cmap=CMAP, norm=LogNorm(vmin=1, vmax=H.max())
    )
    fig = ax.get_figure()
    cbar = fig.colorbar(cmp, ax=ax, orientation="horizontal")
    cbar.set_label(r"Counts")

    return ax


def _fmt_ax(ax, bounds=None):
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\mathcal{M}_{\rm{src}}$")
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

    prob = np.array(
        [prob_fn(mc, z) for mc, z in zip(mc_grid.ravel(), z_grid.ravel())]
    )
    prob = prob.reshape(mc_grid.shape)

    norm = None
    if logscale:
        norm = _get_norm(prob)

    cmp = ax.pcolor(
        z_grid,
        mc_grid,
        prob,
        cmap=CMAP,
        # edgecolors='white',
        # norm=LogNorm(vmin=0.001)
        norm=norm,
    )

    # add colorbar above the axes
    fig = ax.get_figure()
    cbar = fig.colorbar(cmp, ax=ax, orientation="horizontal")
    cbar.set_label(r"$\pi(\mathcal{M}_{\rm{src}},z)$")
    return ax


def _get_norm(x):
    log_x = np.log(x)
    log_x = log_x[np.isfinite(log_x)]
    if len(log_x) == 0:
        return None
    vmin, vmax = np.exp(log_x.min()), x.max()
    # return LogNorm(vmin=np.exp(log_x.min()), vmax=x.max())
    return PowerNorm(gamma=0.3, vmin=vmin / 10, vmax=vmax * 3)


def plot_weights(weights: np.ndarray, mc_bins, z_bins, ax=None, contour=True):
    if ax is None:
        fig, ax = plt.subplots()
    _fmt_ax(
        ax, {Mc: [min(mc_bins), max(mc_bins)], Z: [min(z_bins), max(z_bins)]}
    )
    cmp = ax.pcolor(
        z_bins, mc_bins, weights.T, cmap=CMAP, norm=_get_norm(weights)
    )

    if contour:
        Zb, MCb = np.meshgrid(z_bins, mc_bins)
        ax.contour(
            Zb,
            MCb,
            weights.T,
            levels=1,
            colors="tab:orange",
            linewidths=[0, 2],
            alpha=0.1,
        )
    # add colorbar above the axes
    fig = ax.get_figure()
    cbar = fig.colorbar(cmp, ax=ax, orientation="horizontal")
    cbar.set_label(r"$w_{z,\mathcal{M}}$")
    return ax


def plot_scatter(samples, bounds=None, ax=None, color=CTOP):
    if ax is None:
        fig, ax = plt.subplots()
    _fmt_ax(ax, bounds)
    z, mc = samples[:, 0], samples[:, 1]
    ax.plot(z, mc, marker=".", c=color, lw=0)
    return ax


def add_cntr(ax, X, Y, Z, color=CTOP):
    ax.contour(
        X,
        Y,
        gaussian_filter(Z, 1.2).T,
        levels=1,
        colors=color,
        linewidths=[0, 2],
        alpha=0.1,
    )


def plot_event_mcz_uncertainty(data: pd.DataFrame, pass_fail=None):
    n_events = len(data)
    # sort data by name
    data = data.sort_values("Name", ascending=False)
    z, zup, zlow = (
        data["redshift"],
        data["redshift_plus"],
        data["redshift_minus"],
    )
    mc, mcup, mclow = (
        data["srcmchirp"],
        data["srcmchirp_plus"],
        data["srcmchirp_minus"],
    )
    names = data["Name"]

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(4.5, 0.25 * n_events))
    # plot data in reverse order to match the order of the data
    y = range(n_events)
    kwgs = dict(capsize=4, fmt=",", lw=2, elinewidth=2, markeredgewidth=2)

    # names along the y-axis
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    # pastro along the y-axis on the right side for axes[1] (2dp)
    # twinx() to create a second y-axis
    ax_past = axes[1].twinx()
    ax_past.yaxis.tick_right()
    ax_past.set_yticks(y)
    # format to {0.00} (2dp)
    pastro = data["Pastro"]
    # replace values below 0 --> 0
    pastro[pastro < 0] = 0
    ax_past.set_yticklabels([f"{c:.2f}" for c in pastro])
    # set tick width to 0
    ax_past.tick_params(axis="y", length=0)
    if pass_fail is None:
        pass_fail = [True if _pi >= 0.95 else False for _pi in pastro]
    colors = ["tab:green" if p else "tab:red" for p in pass_fail]

    zerr = np.array([zlow, zup]).T
    mcerr = np.array([mclow, mcup]).T

    for i in range(n_events):
        clr = dict(color=colors[i])
        axes[0].errorbar(
            z[i], y[i], xerr=zerr[i].reshape(-1, 1), **kwgs, **clr
        )
        axes[1].errorbar(
            mc[i], y[i], xerr=mcerr[i].reshape(-1, 1), **kwgs, **clr
        )

    axes[0].set_xlabel(r"$z$")
    axes[1].set_xlabel(r"$\mathcal{M}_{\rm src}\ [M_{\odot}]$")

    # remove whitespace between subplots
    axes[0].set_ylim(-0.5, n_events - 0.5)
    axes[0].set_xlim(0, 1)
    axes[1].set_xlim(0, max(mcup + mc))

    # set xticks at 3 places manually (0, 0.5, 0.8), (5, 30, 70)
    axes[0].set_xticks([0, 0.5, 0.8])
    axes[1].set_xticks([5, 30, 70])

    # set ytick len= 0
    for i in range(2):
        axes[i].tick_params(axis="y", length=0)

    plt.subplots_adjust(wspace=0.00)

    axes[0].annotate(
        "Events",
        xy=(0, 1),
        xytext=(-50, 5),
        xycoords=("axes fraction", "axes fraction"),
        textcoords="offset points",
        ha="center",
        va="bottom",
        rotation=0,
        fontsize=12,
        fontweight="bold",
    )
    axes[1].annotate(
        "P(BBH)",
        xy=(1, 1),
        xytext=(+5, 5),
        xycoords=("axes fraction", "axes fraction"),
        textcoords="offset points",
        ha="center",
        va="bottom",
        rotation=0,
        fontsize=12,
        fontweight="bold",
    )

    return fig, axes


def _color_errorbars(errorbars, colors, cmap="Blues"):
    colors = get_colors_from_cmap(colors, cmap=cmap)

    for line, color in zip(errorbars[0], colors):
        line.set_color(color)
    for cap, color in zip(errorbars[1], np.repeat(colors, 2, axis=0)):
        cap.set_color(color)
