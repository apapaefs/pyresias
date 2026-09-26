"""Rebuild the paper's Herwig comparison panels from the saved histograms.

Run from the repository root:
    MPLBACKEND=Agg python validation/make_comparison_plots.py [-o OUTPUT_DIR]
"""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "reference-results.json").read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-o", "--output", type=Path, default=HERE / "output",
                    help="directory for the PDF panels (default: validation/output)")
OUTPUT = parser.parse_args().output
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 14, "axes.labelsize": 15,
                     "legend.fontsize": 12, "pdf.fonttype": 42})
COLORS = ("#238b45", "#d95f02")


def arrays(sample, observable):
    h = sample["histograms"][observable]
    edges = np.asarray(h["edges"])
    values = np.asarray(h["density"])
    errors = np.asarray(h["standard_error"])
    labels = None
    if observable in ("ng", "nq"):
        assert h["underflow"] == h["overflow"] == h["nonfinite"] == 0
        assert np.allclose(np.diff(edges), 1.)
        counts = np.rint(values * sample["events"]).astype(int)
        assert counts.sum() == sample["events"]
        if observable == "ng":
            counts = np.r_[counts[:16], counts[16:].sum()]
            labels = [str(i) for i in range(16)] + [r"$\geq16$"]
        else:
            assert counts[:2].sum() == counts[3] == counts[5] == 0
            counts = np.array([counts[2], counts[4], counts[6:].sum()])
            labels = ["2", "4", r"$\geq6$"]
        values = counts / sample["events"]
        errors = np.sqrt(values * (1.-values) / (sample["events"]-1))
        edges = np.arange(len(values)+1) - .5
    return edges, values, errors, labels


for observable, xlabel, ylabel, log in (
        ("ptg", r"Gluon $p_T$ [GeV]", r"$(1/N_g)\,dN_g/dp_T$ [GeV$^{-1}$]", True),
        ("yg", "Gluon rapidity", r"$(1/N_g)\,dN_g/dy$", False),
        ("ng", "Final-state gluon count", "Probability per event", False),
        ("nq", "Final-state quark + antiquark count", "Probability per event", True)):
    fig, (ax, ratio) = plt.subplots(2, 1, figsize=(5.4, 4.7), sharex=True,
                                   gridspec_kw={"height_ratios": (3, 1)},
                                   layout="constrained")
    saved = []
    for sample, color in zip(DATA["samples"], COLORS):
        edges, values, errors, labels = arrays(sample, observable)
        centers = (edges[1:]+edges[:-1])/2.
        ax.stairs(values, edges, baseline=None, label=sample["label"],
                  color=color, linewidth=1.5)
        ax.fill_between(edges, np.r_[values-errors, (values-errors)[-1]],
                        np.r_[values+errors, (values+errors)[-1]],
                        step="post", color=color, alpha=.23, linewidth=0)
        saved.append((values, errors))
    h, he = saved[0]
    p, pe = saved[1]
    valid = h > 0
    rel_h = np.divide(he,h,out=np.full_like(h,np.nan),where=valid)
    rel_p = np.divide(pe,h,out=np.full_like(h,np.nan),where=valid)
    quotient = np.divide(p,h,out=np.full_like(h,np.nan),where=valid)
    ratio.fill_between(edges, np.r_[1-rel_h,(1-rel_h)[-1]],
                       np.r_[1+rel_h,(1+rel_h)[-1]], step="post",
                       color=COLORS[0],alpha=.23,linewidth=0)
    ratio.errorbar(centers[valid],quotient[valid],yerr=rel_p[valid],
                   color=COLORS[1],fmt=".",markersize=4,linewidth=1)
    ratio.axhline(1.,color="black",linestyle="--",linewidth=.8)
    ratio.set_ylabel("Pyr./HW")
    ratio.set_xlabel(xlabel)
    ratio.set_xlim(edges[0],edges[-1])
    lo=min(np.nanmin(1-rel_h),np.nanmin(quotient-rel_p))
    hi=max(np.nanmax(1+rel_h),np.nanmax(quotient+rel_p))
    margin=max(.02,.12*(hi-lo))
    ratio.set_ylim(lo-margin,hi+margin)
    if observable == "ng":
        ticks=[0,4,8,12,16]
        ratio.set_xticks(ticks, [labels[t] for t in ticks])
    elif labels:
        ratio.set_xticks(centers,labels)
    if log:
        ax.set_yscale("log")
        positive = [v[v>0] for v,_ in saved]
        ax.set_ylim(min(x.min() for x in positive)*.45,max(x.max() for x in positive)*2)
    else:
        ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    for axis in (ax,ratio):
        axis.tick_params(direction="in",top=True,right=True)
    fig.savefig(OUTPUT / f"full-{observable}.pdf")
    plt.close(fig)
