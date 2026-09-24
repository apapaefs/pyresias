"""Step-by-step Sudakov veto tutorial; no event generation occurs on import."""
import argparse
from pathlib import Path
from random import random, seed
import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from alphaS import alphaS as RunningCoupling, CF
from shower_cli import positive, nonnegative
from simplehisto import simplehisto

debug = False
Q, Qc = 1000., 1.
fixedScale = Q / 2.
scaleoption = "fixed"
tMethod = "Overestimated"
aS = RunningCoupling(.118, 91.1876)


def Pqq(z):
    return CF * (1. + z*z) / (1. - z)


def Pqq_over(z):
    return 2. * CF / (1. - z)


def scale_of_alphaS(t, z):
    return fixedScale if scaleoption == "fixed" else z * (1. - z) * math.sqrt(t)


def alphaS(t, z, Qcut, aSover):
    return aS.alphasQ(max(scale_of_alphaS(t, z), Qcut)) / (2. * math.pi)


def get_alphaS_over(Q, Qcut):
    scale = max(fixedScale, Qcut) if scaleoption == "fixed" else Qcut
    return aS.alphasQ(scale) / (2. * math.pi)


def tGamma(z, aSover):
    return -2. * aSover * CF * np.log1p(-z)


def inversetGamma(r, aSover):
    return -np.expm1(-r / (2. * CF * aSover))


def zp_over(t, Qcut):
    return 1. - Qcut / math.sqrt(t)


def zm_over(t, Qcut):
    return Qcut / math.sqrt(t)


def Get_zEmission(t, Qcut, R, aSover):
    lower = tGamma(zm_over(t, Qcut), aSover)
    upper = tGamma(zp_over(t, Qcut), aSover)
    return inversetGamma(lower + R * (upper - lower), aSover)


def Get_pTsq(t, z):
    return z*z * (1. - z)**2 * t


def Get_mvirtsq(t, z):
    return z * (1. - z) * t


def Get_tEmission_direct(Q, Qcut, R, aSover):
    if Q <= 2. * Qcut or R <= 0.:
        return Q*Q, None, False, False
    rate = tGamma(zp_over(Q*Q, Qcut), aSover) - tGamma(zm_over(Q*Q, Qcut), aSover)
    t = Q*Q * R**(1. / rate)
    alive = t > 4. * Qcut*Qcut
    return t, None, alive, alive


def Get_tEmission(Q, Qcut, R, aSover):
    """Invert the Sudakov with scale-dependent z bounds by numerical integration."""
    if Q <= 2. * Qcut or R <= 0.:
        return Q*Q, None, False, False
    logmin = math.log(4. * Qcut*Qcut / (Q*Q))

    def integrand(logt):
        t = Q*Q * math.exp(logt)
        return tGamma(zp_over(t, Qcut), aSover) - tGamma(zm_over(t, Qcut), aSover)

    def equation(logt):
        return quad(integrand, logt, 0., epsabs=1.e-10)[0] + math.log(R)

    if equation(logmin) <= 0.:
        return Q*Q, None, False, False
    logt = brentq(equation, logmin, 0., xtol=1.e-12)
    return Q*Q * math.exp(logt), None, True, True


def Generate_Emission(Q, Qcut, aSover):
    r1, r2, r3, r4 = random(), random(), random(), random()
    sampler = Get_tEmission_direct if tMethod == "Overestimated" else Get_tEmission
    t, _, alive, _ = sampler(Q, Qcut, r1, aSover)
    if not alive:
        return t, 1., 0., 0., False, False
    # The analytic proposal freezes the z interval at the starting scale.
    # The numerical inversion instead integrates its changing interval.
    proposal_t = Q*Q if tMethod == "Overestimated" else t
    z = Get_zEmission(proposal_t, Qcut, r2, aSover)
    probability = alphaS(t, z, Qcut, aSover) / aSover
    if not 0. <= probability <= 1. + 1.e-14:
        raise RuntimeError("Invalid coupling veto bound")
    accepted = (zm_over(t, Qcut) <= z <= zp_over(t, Qcut)
                and Get_pTsq(t, z) >= Qcut*Qcut
                and r3 <= Pqq(z) / Pqq_over(z) and r4 <= probability)
    if debug:
        print(f"trial sqrt(t)={math.sqrt(t):.6g}, z={z:.6g}, accepted={accepted}")
    if not accepted:
        return t, 1., 0., 0., False, True
    return t, z, Get_pTsq(t, z), Get_mvirtsq(t, z), True, True


def Evolve(Q, Qmin, aSover):
    emissions = []
    t, z = Q*Q, 1.
    while math.sqrt(t) * z > 2. * Qmin:
        t, z, pt2, mass2, accepted, alive = Generate_Emission(math.sqrt(t)*z, Qmin, aSover)
        if not alive:
            break
        if accepted:
            emissions.append([math.sqrt(t), z, math.sqrt(pt2), math.sqrt(mass2)])
    return emissions


def sample_splitting(n, zmin=.01, zmax=.99):
    """Check Pqq alone on a fixed interval, without changing shower phase space."""
    if not 0. < zmin < zmax < 1.:
        raise ValueError("Expected 0 < zmin < zmax < 1")
    result = []
    while len(result) < n:
        z = 1. - (1. - zmin) * ((1. - zmax) / (1. - zmin))**random()
        if random() <= Pqq(z) / Pqq_over(z):
            result.append(z)
    return np.asarray(result)


def splitting_figure(values, path=None, zmin=.01, zmax=.99, weighted=True):
    """Normalized shape and bin-integrated reference on identical fixed support."""
    from matplotlib import pyplot as plt
    edges = np.linspace(zmin, zmax, 41)
    values = np.asarray(values)
    weights = 1. - values if weighted else np.ones_like(values)
    counts = np.histogram(values, edges, weights=weights)[0]
    variance = np.histogram(values, edges, weights=weights**2)[0]
    norm = weights.sum()
    shape = counts / norm / np.diff(edges) if norm else np.zeros_like(counts)
    error = np.zeros_like(counts)
    if norm and len(values) > 1:
        fraction = counts / norm
        residual = variance * (1. - 2. * fraction) + fraction**2 * np.sum(weights**2)
        error = np.sqrt(np.maximum(0., residual) * len(values)/(len(values)-1)) / norm / np.diff(edges)
    kernel = (lambda z: CF * (1. + z*z)) if weighted else Pqq
    integral = quad(kernel, zmin, zmax)[0]
    reference = np.array([quad(kernel, lo, hi)[0] / integral / (hi-lo)
                          for lo, hi in zip(edges[:-1], edges[1:])])
    centers = (edges[1:] + edges[:-1]) / 2.
    fig, (ax, ratio) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 5),
                                  gridspec_kw={"height_ratios": (3, 1)}, layout="constrained")
    ax.stairs(reference, edges, label="Splitting function", color="blue")
    ax.errorbar(centers, shape, yerr=error, fmt=".", label="Veto samples", color="red")
    ratio.errorbar(centers, shape/reference, yerr=error/reference, fmt=".", color="red")
    ratio.axhline(1., color="black", linestyle="--")
    ax.set_ylabel("Normalized weighted density" if weighted else "Normalized density")
    ratio.set(xlabel="$z$", ylabel="MC/ref.")
    ax.legend()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def main(argv=None):
    global debug, Q, Qc, fixedScale, scaleoption, tMethod
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--nevolve", type=nonnegative, default=1000)
    parser.add_argument("-Q", type=positive, default=1000.)
    parser.add_argument("-c", dest="Qc", type=positive, default=1.)
    parser.add_argument("-o", "--output", type=Path, default=Path("plots"))
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--method", choices=("Overestimated", "Numerical"), default="Overestimated")
    parser.add_argument("--coupling", choices=("fixed", "pt"), default="fixed")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    debug, Q, Qc, tMethod = args.debug, args.Q, args.Qc, args.method
    fixedScale, scaleoption = Q/2., args.coupling
    seed(args.seed)
    try:
        over = get_alphaS_over(Q, Qc)
    except ValueError as exc:
        parser.error(str(exc))
    emissions = []
    for _ in range(args.nevolve):
        emissions.extend(Evolve(Q, Qc, over))
    print(f"Evolved {args.nevolve} branches; {len(emissions)} emissions (seed={args.seed})")
    if not args.no_plots:
        from matplotlib import pyplot as plt
        columns = np.asarray(emissions).reshape(-1, 4).T
        for name, column, label in (("evolutionvar", 0, r"$\sqrt{t}$ [GeV]"),
                                     ("transversemom", 2, r"$p_T$ [GeV]"),
                                     ("virtmass", 3, r"$m_{\mathrm{virt}}$ [GeV]")):
            simplehisto("Plotting " + name, name, args.output, columns[column], label, "Density [GeV$^{-1}$]")
        zs = sample_splitting(args.nevolve)
        for weighted, filename in ((True, "momentumfrac.pdf"), (False, "momentumfrac_p.pdf")):
            fig = splitting_figure(zs, args.output / filename, weighted=weighted)
            plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
