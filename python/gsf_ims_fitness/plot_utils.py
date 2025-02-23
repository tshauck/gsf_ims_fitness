"""
Plotting utilities for the GSF IMS Fitness project.
"""

import seaborn as sns


def plot_colors():
    return sns.hls_palette(12, l=0.4, s=0.8)


def plot_colors96():
    p_c12 = []
    for c in plot_colors():
        for i in range(8):
            p_c12.append(c)
    return p_c12


def plot_colors48():
    p_c12 = []
    for c in plot_colors():
        for i in range(4):
            p_c12.append(c)
    return p_c12
