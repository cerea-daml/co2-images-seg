# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns


def setMatplotlibParam():
    """Set defaults matplotlib parameters for visually nice plots."""

    sns.set_context("paper")
    sns.set_style("whitegrid")

    plt.rc("lines", linewidth=0.66)
    plt.rc("lines", markeredgewidth=0.5)
    plt.rc("lines", markersize=4)
    plt.rc("figure", dpi=300)
    plt.rc("font", family="serif")
    plt.rc("savefig", format="pdf")
    plt.rc("savefig", facecolor="white")
    plt.rc("axes", linewidth=1.2)
    plt.rc("axes", edgecolor="k")
    plt.rc("axes", facecolor=[0.96, 0.96, 0.96])
    plt.rc("axes", labelsize="x-small")
    plt.rc("axes", titlesize="x-small")
    plt.rc("legend", fontsize="x-small")
    plt.rc("legend", frameon=True)
    plt.rc("legend", framealpha=1)
    plt.rc("legend", handlelength=3)
    plt.rc("legend", numpoints=3)
    plt.rc("legend", markerscale=1)
    plt.rc("xtick", labelsize="x-small")
    plt.rc("ytick", labelsize="x-small")
    plt.rc("xtick.major", pad=0)
    plt.rc("ytick.major", pad=0)
    plt.rc('grid', linewidth = 0.6)
    plt.rc('grid', alpha = 0.6)

def setMatplotlibParam_KdePlot():
    """Set matplotlib parameters for nice kde plots."""
    plt.rcdefaults()
    sns.set_context('paper')
    sns.set_style('whitegrid')

    plt.rc('lines',         linewidth           = 0.5)
    plt.rc('font',          family              = 'sans-serif')
    plt.rc('savefig',       facecolor           = 'white')
    plt.rc('axes',          linewidth           = 1.2)
    plt.rc('axes',          edgecolor           = 'k')
    plt.rc('axes',          facecolor           = [0.96, 0.96, 0.96])
    plt.rc('axes',          labelsize           = 15)
    plt.rc('axes',          titlesize           = 15)
    plt.rc('legend',        fontsize            = 10)
    plt.rc('legend',        frameon             = True)
    plt.rc('legend',        framealpha          = 1)
    plt.rc('legend',        handlelength        = 3)
    plt.rc('legend',        numpoints           = 3)
    plt.rc('legend',        markerscale         = 1)
    plt.rc('xtick',         labelsize           = 'x-small')
    plt.rc('ytick',         labelsize           = 'x-small')
    plt.rc('xtick.major',   pad                 = 0)
    plt.rc('ytick.major',   pad                 = 0)

def set_figure_axs(
    nrows: int = 1,
    ncols: int = 1,
    pad_w_ext_left: float = 0.3,
    pad_w_ext_right: float = 0.3,
    pad_w_int: float = 0.35,
    pad_h_ext: float = 0.2,
    pad_h_int: float = 0.33,
    wratio: float = 0.35,
    hratio: float = 0.75,
    projection=None,
    with_defaults_plt=True,
) -> list:
    """Proper setting of figure axs:
    - wratio and hratio control the ratio w/h of each ax
    - pad_[w,h]_ext control the padding on the borders of the figure
    - pad_[w,h]_int control the padding between the diverse axs
    - projection defines a projection for figure if necessary.
    """

    linewidth = 5.80

    ax_w = wratio * linewidth
    ax_h = hratio * ax_w

    fig_w = pad_w_ext_left + pad_w_ext_right + ncols * ax_w + (ncols - 1) * pad_w_int
    fig_h = 2 * pad_h_ext + nrows * ax_h + (nrows - 1) * pad_h_int

    axs_x = np.zeros(ncols)
    for i_col in range(ncols):
        axs_x[i_col] = (pad_w_ext_left + i_col * (ax_w + pad_w_int)) / fig_w

    axs_y = np.zeros(nrows)
#    for i_row in reversed(range(nrows)):
    for i_row in range(nrows):
        axs_y[nrows-1-i_row] = (pad_h_ext + i_row * (ax_h + pad_h_int)) / fig_h

    ax_dx = ax_w / fig_w
    ax_dy = ax_h / fig_h

    figure = plt.figure(figsize=(fig_w, fig_h))
    axs = [None] * nrows * ncols
    for i_row in range(nrows):
        for i_col in range(ncols):
            i_ax = i_row * ncols + i_col
            axs[i_ax] = figure.add_axes(
                [axs_x[i_col], axs_y[i_row], ax_dx, ax_dy], projection=projection
            )

    return axs


# ------------------------------------------------------------------------
# clean_axs_of_violin_plots
def clean_axs_of_violin_plots(axs):

    for ax in axs:
        for index_line, line in zip(range(len(ax.lines)), ax.lines):
            if index_line % 3 == 0:
                line.set_linestyle("--")
                line.set_color("black")
                line.set_alpha(0.9)

            if (index_line + 1) % 3 == 0:
                line.set_linestyle("--")
                line.set_color("black")
                line.set_alpha(0.9)

            if (index_line + 2) % 3 == 0:
                line.set_linestyle("-")
                line.set_color("black")
                line.set_alpha(1)
                line.set_linewidth(1)

    return axs


# ------------------------------------------------------------------------
# set_sns_histplot_legend
def set_sns_histplot_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)


# __________________________________________________________
