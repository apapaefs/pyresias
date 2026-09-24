"""Shared command-line and event handling for the educational showers."""
import argparse
from collections import Counter
from itertools import islice
import json
from pathlib import Path
import random
import sys

import numpy as np
from tqdm import tqdm

from kinematics import GlobalMomCons, KinematicsError
from lhe_io import LHEFile, assign_quark_colors, write_event, lhe_output


def positive(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0.:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return value


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('must be a nonnegative integer')
    return value


def validate_hard_event(particles):
    incoming = [p for p in particles if p[1] == -1]
    outgoing = [p for p in particles if p[1] == 1]
    if sorted(p[0] for p in incoming) != [-11, 11] or len(outgoing) != 2:
        raise ValueError('Only e+e- -> q qbar hard events are supported')
    a, b = outgoing
    if not 1 <= abs(a[0]) <= 5 or a[0] != -b[0]:
        raise ValueError('The hard final state must be one light quark-antiquark pair')
    for p in incoming + outgoing:
        if not np.all(np.isfinite(p[2:7])) or p[5] <= 0.:
            raise ValueError('Hard momenta must be finite with positive energies')
        if abs(p[6]) > 1.e-7 or not np.isclose(np.linalg.norm(p[2:5]), p[5], rtol=1.e-7):
            raise ValueError('The tutorial supports massless hard particles only')
    total = np.sum([p[2:6] for p in incoming], axis=0)
    if np.linalg.norm(total[:3]) > 1.e-7 * total[3]:
        raise ValueError('The input must be in the center-of-mass frame')
    if not np.allclose(np.sum([p[2:6] for p in outgoing], axis=0), total,
                       rtol=0., atol=1.e-7 * total[3]):
        raise ValueError('The hard event does not conserve four-momentum')
    q, qb = (a, b) if a[0] > 0 else (b, a)
    if q[7] <= 0 or q[7] != qb[8] or q[8] != 0 or qb[7] != 0:
        raise ValueError('The hard quark pair must share one color line')


def reconstruct_event(shower, particles, cutoff, overestimate, max_attempts=100,
                      assign_colors=True, statistics=None):
    validate_hard_event(particles)
    if max_attempts < 1:
        raise ValueError('max_attempts must be positive')
    for attempt in range(max_attempts):
        try:
            if statistics is None:
                partons, jets = shower(particles, cutoff, overestimate)
            else:
                statistics.clear()  # Count only the ultimately accepted shower.
                partons, jets = shower(particles, cutoff, overestimate, statistics=statistics)
            if assign_colors:
                assign_quark_colors(jets)
            return GlobalMomCons(partons, jets), attempt
        except KinematicsError:
            # Resample the shower on this same hard event, never drop its weight.
            if attempt == max_attempts - 1:
                raise


def run_shower(module, argv=None, qtilde=False, full=False):
    parser = argparse.ArgumentParser(description=module.__doc__)
    parser.add_argument('inputfile', type=Path)
    parser.add_argument('-n', '--nshower', type=nonnegative, default=None,
                        help='Number of hard events (default: all; zero writes an empty sample)')
    parser.add_argument('-o', '--output', type=Path)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-p', '--printevents', action='store_true')
    parser.add_argument('-c', '--coupling-freeze' if qtilde else '--cutoff',
                        dest='cutoff', type=positive, default=.935)
    if qtilde:
        parser.add_argument('--ptmin', type=positive, default=.900)
    if full:
        parser.add_argument('--flavours', type=int, choices=range(6), default=5,
                            help='Massless g->qqbar flavours, 1=d through 5=b; 0 disables pairs')
        parser.add_argument('--quark-only', action='store_true',
                            help='Disable gluon branching to recover the quark-line example')
    args = parser.parse_args(argv)
    suffix = '_pyr_full.lhe' if full else '_pyr.lhe'
    output = args.output or args.inputfile.with_name(
        args.inputfile.name.removesuffix('.gz').removesuffix('.lhe') + suffix)
    if output.resolve() == args.inputfile.resolve():
        parser.error('Input and output paths must differ')
    if output.suffix != '.lhe':
        parser.error('The shower writes LHE; choose an output name ending in .lhe')
    module.debug = args.debug
    module.Qc = args.cutoff
    if qtilde:
        module.pTmin = args.ptmin
    if full:
        module.Nflavours = args.flavours
        module.gluon_splittings = not args.quark_only
    random.seed(args.seed)
    try:
        over = module.get_alphaS_over(args.cutoff)
        output.parent.mkdir(parents=True, exist_ok=True)
        count = retries = 0
        branches = Counter()
        event_branches = {} if full else None
        with LHEFile(args.inputfile) as source, lhe_output(output, source.preamble) as out:
            out.write(f'<!-- Pyresias {module.__name__}; seed={args.seed}; '
                      f'coupling cutoff={args.cutoff}; emission cutoff='
                      f'{args.ptmin if qtilde else args.cutoff} GeV -->\n')
            if full:
                out.write(f'<!-- massless flavours={args.flavours}; '
                          f'gluon splittings={module.gluon_splittings}; CMW={module.CMW} -->\n')
            for event in tqdm(islice(source, args.nshower), total=args.nshower,
                              disable=not sys.stderr.isatty()):
                showered, attempts = reconstruct_event(
                    module.Shower, event.particles, args.ptmin if qtilde else args.cutoff, over,
                    assign_colors=not full, statistics=event_branches)
                retries += attempts
                if full:
                    branches.update(event_branches)
                if args.debug or args.printevents:
                    module.PrintMomenta(showered)
                write_event(out, showered, event)
                count += 1
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'{parser.prog}: {exc}\n')
    print(f'Wrote {count} events to {output} (seed={args.seed}, reconstruction retries={retries})')
    if full:
        print('Accepted branchings: ' + json.dumps(dict(sorted(branches.items()))))
    return 0
