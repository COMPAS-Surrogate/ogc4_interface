import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import stats
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, colorConverter
import seaborn as sns



SIGMA_LEVELS = [1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)]
SIGMA_LEVELS = [SIGMA_LEVELS[0], SIGMA_LEVELS[-1]] # 3 and 1-sigma levels


def format_cntr_data(x ,y, levels, smooth=1.5, bins=100, range=[[0.05 ,1] ,[0 ,0.6]]):
    H, X, Y = np.histogram2d(
        x.flatten(),
        y.flatten(),
        bins=bins,
        range=range,
    )

    if H.sum() == 0:
        raise ValueError(
            "It looks like the provided 'range' is not valid "
            "or the sample is empty."
        )

    if smooth is not None:
        if gaussian_filter is None:
            raise ImportError("Please install scipy for smoothing")
        H = gaussian_filter(H, smooth)


    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except IndexError:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m):
        raise ValueError("Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate(
        [
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
            ]
    )
    Y2 = np.concatenate(
        [
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
            ]
    )
    return X2, Y2, H2, V, H




def add_cntr(
        ax, x ,y, color, label=None, levels=SIGMA_LEVELS, labelypos=0, bin_range=[[0. ,1] ,[0 ,40]],
        fill=False):
    cmap = sns.color_palette(color, as_cmap=True)
    c = cmap(0.75)

    # LINE
    X2, Y2, H2, V, H = format_cntr_data(x ,y, levels=[levels[-1]], range=bin_range)
    ax.contour(X2, Y2, H2.T, V, colors=[c] * len(levels), alpha=0.25, antialiased=True)

    if fill:
        # FILL
        X2, Y2, H2, V, H = format_cntr_data(x ,y, levels=levels, range=bin_range)
        ax.contourf(
            X2,
            Y2,
            H2.T,
            [V.min(), H.max()],
            cmap=cmap,
            antialiased=False,
            alpha=0.05,
        )
        ax.contourf(
            X2,
            Y2,
            H2.T,
            [V.max(), H.max()],
            cmap=cmap,
            antialiased=False,
            alpha=0.25
        )
    if label:
        ax.plot(np.median(x), np.median(y), color=c, zorder=100, alpha=0.5, label=label)
        med_x, xup, xlow = get_median_and_err(x)
        med_y, yup, ylow = get_median_and_err(y)
        # ax.errorbar(med_x, med_y, xerr=[[xlow], [xup]],  yerr=[[ylow], [yup]], color=c, zorder=100, alpha=0.1)
        ax.scatter(med_x, med_y, color=c, zorder=100, alpha=1)
        print(f"{label}: q {median_val_and_1_sig_err(x)}, xeff {get_str_for_val(y)}")
        add_txt_marker(ax, med_x, med_y, c, label, labelypos)
