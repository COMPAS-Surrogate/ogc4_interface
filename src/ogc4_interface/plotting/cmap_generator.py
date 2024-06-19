import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.colors import (to_rgba, ListedColormap)
from scipy.interpolate import interp1d
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, sRGBColor


def lab_to_rgb(*args):
    """Convert Lab color to sRGB, with components clipped to (0, 1)."""
    Lab = LabColor(*args)
    sRGB = convert_color(Lab, sRGBColor)
    return np.clip(sRGB.get_value_tuple(), 0, 1)


def get_cylon():
    L_samples = np.linspace(100, 0, 5)

    a_samples = (
        33.34664938,
        98.09940562,
        84.48361516,
        76.62970841,
        21.43276891)

    b_samples = (
        62.73345997,
        2.09003022,
        37.28252236,
        76.22507582,
        16.24862535)

    L = np.linspace(100, 0, 255)
    a = interp1d(L_samples, a_samples[::-1], 'cubic')(L)
    b = interp1d(L_samples, b_samples[::-1], 'cubic')(L)

    colors = [lab_to_rgb(Li, ai, bi) for Li, ai, bi in zip(L, a, b)]
    cmap = np.vstack(colors)
    return ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])


def get_top_color_of_cmap(cmap):
    return cmap(np.linspace(0, 1, 256))[-1]

CMAP = get_cylon()
CTOP = get_top_color_of_cmap(CMAP)
