"""Density histograms used by the introductory tutorial."""
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np


def simplehisto(message, plot_type, outputdirectory, array_to_hist, xlab, ylab,
                nbins=50, xlog=False, ylog=False, norm=True):
    print(message)
    values = np.asarray(array_to_hist, dtype=float)
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=nbins)
    heights = counts.astype(float)
    if norm and counts.sum():
        heights /= counts.sum() * np.diff(edges)
    fig, ax = plt.subplots(layout="constrained")
    ax.stairs(heights, edges, color="red")
    ax.set(xlabel=xlab, ylabel=ylab)
    if xlog:
        ax.set_xscale("log")
    if ylog and np.any(heights > 0):
        ax.set_yscale("log")
    if not values.size:
        ax.text(.5, .5, "No emissions", ha="center", transform=ax.transAxes)
    output = Path(outputdirectory)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / (plot_type + ".pdf"))
    plt.close(fig)
