"""Sudakov closure test of the single-line veto sampler in pyresias_test.py.

The tutorial's Evolve function is run for many independent quark lines with the
running coupling. The scale of the first accepted emission of each line is
compared with the prediction obtained by integrating the branching rate Gamma(t)
numerically, including the probability of reaching the cutoff without emission.

Run from the repository root, with the tutorial dependencies installed:
    MPLBACKEND=Agg python validation/make_sudakov_plot.py [-o OUTPUT_DIR]
"""
import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.stats import chi2 as chi2_distribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("code", type=Path, nargs="?", default=repo,
                        help="Pyresias code repository (default: this repository)")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path(__file__).resolve().parent / "output",
                        help="directory for the PDF and JSON (default: validation/output)")
    parser.add_argument("-n", "--lines", type=int, default=5000000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("-Q", type=float, default=206.)
    parser.add_argument("-c", dest="Qc", type=float, default=1.)
    parser.add_argument("--bins", type=int, default=40)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code.resolve()))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import pyresias_test as tutorial

    # Configure the module as its command line does, with the running coupling.
    Q, Qc = args.Q, args.Qc
    tutorial.Q, tutorial.Qc, tutorial.fixedScale = Q, Qc, Q/2.
    tutorial.scaleoption, tutorial.tMethod, tutorial.debug = "pt", "Overestimated", False
    over = tutorial.get_alphaS_over(Q, Qc)

    # Monte Carlo: first accepted emission of each line.
    random.seed(args.seed)
    start = time.time()
    first, none = [], 0
    for _ in range(args.lines):
        emissions = tutorial.Evolve(Q, Qc, over)
        if emissions:
            first.append(emissions[0][0])  # sqrt(t) of the first emission
        else:
            none += 1
    runtime = time.time() - start

    # Reference: t*Gamma(t) on the physical region z(1-z)sqrt(t) >= Qc.
    def t_gamma(t):
        root = 1. - 4.*Qc/math.sqrt(t)
        if root <= 0.:
            return 0.
        zm, zp = (1. - math.sqrt(root))/2., (1. + math.sqrt(root))/2.
        return quad(lambda z: tutorial.alphaS(t, z, Qc, over)*tutorial.Pqq(z),
                    zm, zp, epsrel=1.e-10, limit=200)[0]

    edges = np.geomspace(4.*Qc, Q, args.bins + 1)  # in sqrt(t)
    exponent = np.zeros_like(edges)  # integral of Gamma from edge**2 to Q**2
    for k in range(args.bins - 1, -1, -1):
        exponent[k] = exponent[k+1] + quad(lambda u: t_gamma(math.exp(u)),
                                           2.*math.log(edges[k]),
                                           2.*math.log(edges[k+1]),
                                           epsrel=1.e-10, limit=200)[0]
    sudakov = np.exp(-exponent)
    p_none = float(sudakov[0])  # no emission above the threshold t = 16 Qc^2
    p_bins = np.diff(sudakov)

    # Multinomial comparison: the bins plus the no-emission category.
    counts = np.histogram(first, edges)[0]
    assert counts.sum() + none == args.lines
    observed = np.r_[none, counts]
    expected = args.lines*np.r_[p_none, p_bins]
    chi2 = float(np.sum((observed - expected)**2/expected))
    ndf = len(observed) - 1
    mc_none = none/args.lines
    mc_none_error = math.sqrt(mc_none*(1. - mc_none)/args.lines)

    # Figure: probability per unit ln(sqrt t) for the first emission.
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 15,
                         "legend.fontsize": 12, "pdf.fonttype": 42})
    width = np.diff(np.log(edges))
    density = counts/args.lines/width
    error = np.sqrt(counts*(1. - counts/args.lines))/args.lines/width
    reference = p_bins/width
    centers = np.sqrt(edges[1:]*edges[:-1])
    fig, (ax, ratio) = plt.subplots(2, 1, figsize=(5.4, 4.7), sharex=True,
                                    gridspec_kw={"height_ratios": (3, 1)},
                                    layout="constrained")
    ax.stairs(reference, edges, baseline=None, color="black", linewidth=1.2,
              label=r"$\Gamma(t)\,\Delta(t,t_{\max})$, numerical")
    ax.errorbar(centers, density, yerr=error, fmt=".", color="#d95f02",
                markersize=5, linewidth=1, label="Veto algorithm")
    ax.text(.04, .70, "No emission:\n"
            rf"MC ${mc_none:.5f}\pm{mc_none_error:.5f}$" "\n"
            rf"$\Delta(16Q_c^2,t_{{\max}})={p_none:.5f}$",
            transform=ax.transAxes, fontsize=11, va="top")
    ratio.errorbar(centers, density/reference, yerr=error/reference, fmt=".",
                   color="#d95f02", markersize=5, linewidth=1)
    ratio.axhline(1., color="black", linestyle="--", linewidth=.8)
    ratio.set_xscale("log")
    ratio.set_xlim(edges[0], edges[-1])
    spread = np.nanmax(np.abs(density/reference - 1.) + error/reference)
    ratio.set_ylim(1. - max(.02, 1.15*spread), 1. + max(.02, 1.15*spread))
    ratio.set_xlabel(r"First-emission scale $\sqrt{t}$ [GeV]")
    ratio.set_ylabel("MC/num.")
    ax.set_ylabel(r"$d\mathcal{P}/d\ln\sqrt{t}$ per line")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    for axis in (ax, ratio):
        axis.tick_params(direction="in", top=True, right=True, which="both")
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "sudakov-closure.pdf"
    fig.savefig(output)
    plt.close(fig)

    try:
        commit = subprocess.run(["git", "-C", str(args.code), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    summary = {
        "description": "First accepted emission of single quark lines from pyresias_test.Evolve",
        "settings": {"Q_GeV": Q, "Qc_GeV": Qc, "coupling": "running, mu=max(pT,Qc)",
                     "alphaS_overestimate": over*2.*math.pi, "method": "Overestimated",
                     "lines": args.lines, "seed": args.seed, "bins": args.bins},
        "pyresias_commit": commit,
        "no_emission": {"monte_carlo": mc_none, "standard_error": mc_none_error,
                        "numerical": p_none},
        "pearson_chi2": chi2, "ndf": ndf,
        "p_value": float(chi2_distribution.sf(chi2, ndf)),
        "edges_sqrt_t_GeV": edges.tolist(), "counts": counts.tolist(),
        "reference_probabilities": p_bins.tolist(),
        "runtime_seconds": runtime,
    }
    (args.output / "sudakov-closure.json").write_text(
        json.dumps(summary, indent=1) + "\n")
    print(f"{output}\nno emission: MC {mc_none:.5f} +- {mc_none_error:.5f}, "
          f"numerical {p_none:.5f}; chi2/ndf = {chi2:.1f}/{ndf} "
          f"(p = {summary['p_value']:.3f}); {runtime:.0f} s")


if __name__ == "__main__":
    main()
