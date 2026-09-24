"""Compare final-state LHE distributions with consistent bins and event-level errors."""
import argparse
from itertools import islice
import json
from pathlib import Path
import sys

import numpy as np
from matplotlib import pyplot as plt

# Allow both "python analysis/lhe_analyzer.py" and module imports from the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lhe_io import LHEFile
from shower_cli import nonnegative

BINS = {
    'ng': np.arange(-.5, 40.5), 'nq': np.arange(-.5, 12.5),
    'Eg': np.arange(0., 210., 2.), 'Eq': np.arange(0., 120., 2.),
    'ptg': np.arange(0., 110., 5.), 'pt': np.arange(0., 120., 5.),
    'yg': np.linspace(-3., 3., 50), 'yq': np.linspace(-3., 3., 50),
    'pzg': np.arange(-120., 122., 2.), 'pzq': np.arange(-120., 125., 5.),
    'minvg': np.linspace(-1.e-6, 1.e-6, 51),
    'costheta': np.linspace(-1., 1., 41),
    'sumpvecmag': np.linspace(0., 1.e-5, 51),
    'njets': np.arange(-.5, 20.5),
    'ptj': np.arange(0., 120., 5.), 'ptj1': np.arange(0., 120., 5.),
    'ptj2': np.arange(0., 120., 5.),
}
LABELS = {
    'ng': 'Number of final-state gluons', 'nq': 'Number of final-state quarks',
    'Eg': 'Gluon energy [GeV]', 'Eq': 'Quark energy [GeV]',
    'ptg': r'Gluon $p_T$ [GeV]', 'pt': r'Quark $p_T$ [GeV]',
    'yg': 'Gluon rapidity', 'yq': 'Quark rapidity',
    'pzg': r'Gluon $p_z$ [GeV]', 'pzq': r'Quark $p_z$ [GeV]',
    'minvg': r'Gluon mass squared [GeV$^2$]',
    'costheta': r'Quark $\cos\theta$', 'sumpvecmag': r'$|\sum_{\mathrm{final}}\mathbf{p}|$ [GeV]',
    'njets': 'Number of anti-$k_T$ jets ($R=0.4$)',
    'ptj': r'Jet $p_T$ [GeV]', 'ptj1': r'Leading jet $p_T$ [GeV]',
    'ptj2': r'Subleading jet $p_T$ [GeV]',
}
JET_KEYS = {'njets', 'ptj', 'ptj1', 'ptj2'}


class Histogram:
    """Moments of event contributions, including normalization covariance."""
    def __init__(self, edges, normalization='shape'):
        self.edges = np.asarray(edges)
        self.normalization = normalization
        self.n = 0
        self.counts = np.zeros(len(edges)-1)
        self.squares = self.counts.copy()
        self.cross = self.counts.copy()
        self.population = self.population2 = 0.
        self.underflow = self.overflow = self.nonfinite = 0

    def add(self, values, weight=1.):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(weight):
            raise ValueError('Nonfinite event weight')
        counts = np.histogram(values[np.isfinite(values)], self.edges)[0] * weight
        population = weight * (len(values) if self.normalization == 'shape' else 1)
        self.n += 1
        self.counts += counts
        self.squares += counts**2
        self.cross += counts * population
        self.population += population
        self.population2 += population**2
        self.underflow += int(np.sum(values < self.edges[0]))
        self.overflow += int(np.sum(values > self.edges[-1]))
        self.nonfinite += int(np.sum(~np.isfinite(values)))

    def result(self):
        defined = self.population != 0.
        if not defined and np.any(self.counts):
            raise ValueError('Event weights cancel: normalized distribution is undefined')
        mean = self.counts / self.population if defined else np.zeros_like(self.counts)
        error = np.zeros_like(mean)
        if defined and self.n > 1:
            residual = self.squares - 2*mean*self.cross + mean*mean*self.population2
            error = np.sqrt(np.maximum(0., residual) * self.n/(self.n-1)) / abs(self.population)
        width = np.diff(self.edges)
        return {'edges': self.edges.tolist(), 'density': (mean/width).tolist(),
                'standard_error': (error/width).tolist(), 'events': self.n,
                'normalization': self.normalization, 'population': self.population,
                'normalization_defined': defined,
                'uncertainty_available': self.n > 1 and defined,
                'underflow': self.underflow, 'overflow': self.overflow,
                'nonfinite': self.nonfinite}


def event_observables(particles, jets=False):
    final = [p for p in particles if p[1] == 1]
    if any(not np.all(np.isfinite(p[2:7])) or p[5] <= 0. for p in final):
        raise ValueError('Nonfinite or nonpositive final-state momentum')
    quarks = [p for p in final if 1 <= abs(p[0]) <= 5]
    gluons = [p for p in final if p[0] == 21]
    result = {'ng': [len(gluons)], 'nq': [len(quarks)],
              'sumpvecmag': [float(np.linalg.norm(np.sum([p[2:5] for p in final], axis=0)))]}
    for kind, group in (('g', gluons), ('q', quarks)):
        array = np.asarray([p[2:6] for p in group], dtype=float).reshape(-1, 4)
        px, py, pz, energy = array.T
        with np.errstate(divide='ignore', invalid='ignore'):
            rapidity = .5 * np.log((energy + pz)/(energy - pz))
        result['E'+kind] = energy
        result['ptg' if kind == 'g' else 'pt'] = np.hypot(px, py)
        result['pz'+kind] = pz
        result['y'+kind] = rapidity
        if kind == 'g':
            result['minvg'] = energy**2 - px**2 - py**2 - pz**2
        else:
            result['costheta'] = pz / np.linalg.norm(array[:, :3], axis=1)
    if jets:
        import fastjet
        partons = [fastjet.PseudoJet(*p[2:6]) for p in quarks + gluons]
        cluster = fastjet.ClusterSequence(partons, fastjet.JetDefinition(fastjet.antikt_algorithm, .4))
        clustered = fastjet.sorted_by_pt(cluster.inclusive_jets())
        pts = [p.perp() for p in clustered]
        result.update(njets=[len(pts)], ptj=pts, ptj1=pts[:1], ptj2=pts[1:2])
    return result


def analyze_file(path, keys, limit=None, normalization='shape', jets=False):
    histograms = {key: Histogram(BINS[key], normalization) for key in keys}
    count = 0
    with LHEFile(path) as source:
        for event in islice(source, limit):
            data = event_observables(event.particles, jets=jets)
            for key, histogram in histograms.items():
                histogram.add(data[key], event.weight)
            count += 1
    return {'path': str(path), 'events': count,
            'histograms': {key: histogram.result() for key, histogram in histograms.items()}}


def plot_comparison(samples, labels, key, output):
    comparison = len(samples) > 1
    if comparison:
        fig, (ax, ratio) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 5.),
                                       gridspec_kw={'height_ratios': (3, 1)}, layout='constrained')
    else:
        fig, ax = plt.subplots(layout='constrained')
        ratio = None
    reference = np.asarray(samples[0]['histograms'][key]['density'])
    for sample, label in zip(samples, labels):
        hist = sample['histograms'][key]
        y, err, edges = (np.asarray(hist[name]) for name in ('density', 'standard_error', 'edges'))
        line = ax.stairs(y, edges, label=label)
        ax.fill_between(edges, np.r_[y-err, (y-err)[-1]], np.r_[y+err, (y+err)[-1]],
                        step='post', color=line.get_edgecolor(), alpha=.2, linewidth=0)
        if ratio is not None and sample is not samples[0]:
            values = np.divide(y, reference, out=np.full_like(y, np.nan), where=reference != 0)
            ratio.stairs(values, edges, color=line.get_edgecolor(), label=label)
    ax.set_ylabel('Density' if hist['normalization'] == 'shape' else 'Weighted entries/event/bin width')
    (ratio if comparison else ax).set_xlabel(LABELS[key])
    if comparison:
        ratio.axhline(1., linestyle='--', color='black', linewidth=1)
        ratio.set_ylabel('Ratio to first')
    densities = np.asarray([s['histograms'][key]['density'] for s in samples])
    if key in ('ptg', 'Eg') and np.all(densities >= 0) and np.any(densities > 0):
        ax.set_yscale('log')
    ax.legend(frameon=False)
    fig.savefig(output / (key + '.pdf'))
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('-o', '--output', type=Path, default=Path('plots'))
    parser.add_argument('-n', '--max-events', type=nonnegative)
    parser.add_argument('--labels', nargs='+', help='One legend label per input (default: filenames)')
    parser.add_argument('--normalization', choices=('shape', 'per-event'), default='shape')
    parser.add_argument('--jets', action='store_true', help='Also cluster final-state partons with FastJet')
    parser.add_argument('--observables', nargs='+', choices=list(BINS))
    args = parser.parse_args(argv)
    keys = args.observables or [key for key in BINS if args.jets or key not in JET_KEYS]
    if not args.jets and any(key in JET_KEYS for key in keys):
        parser.error('Jet observables require --jets')
    labels = args.labels or [path.name for path in args.inputs]
    if len(labels) != len(args.inputs):
        parser.error('Supply one label per input')
    try:
        samples = [analyze_file(path, keys, args.max_events, args.normalization, args.jets)
                   for path in args.inputs]
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f'{parser.prog}: {exc}\n')
    args.output.mkdir(parents=True, exist_ok=True)
    for key in keys:
        plot_comparison(samples, labels, key, args.output)
    summary = {'samples': samples, 'labels': labels,
               'uncertainties': 'Per-sample event-level standard errors including normalization covariance. Ratios have no error band; between-sample covariance is not estimated.',
               'jets': 'anti-kt, R=0.4, all final-state light quarks and gluons' if args.jets else None}
    (args.output / 'histograms.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Wrote {len(keys)} comparison plots to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
