"""Independent Sudakov probabilities and invariants of the branching tree."""
from collections import Counter
import contextlib
from io import StringIO
from itertools import islice
import math
from pathlib import Path
import random
import sys
from unittest.mock import patch

import numpy as np
import pytest
from scipy.integrate import quad

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import pyresias_qtilde_full as full
from kinematics import KinematicsError
import shower_cli
import pyresias_qtilde as quark_only
from shower_cli import reconstruct_event
from lhe_io import LHEFile, lhe_output, write_event

INPUT = REPO / 'data/eejj_ECM206_1E6.lhe.gz'


@pytest.mark.parametrize('channel,kernel,over', [('qg', full.Pqq, full.Pqq_over),
         ('gg', full.Pgg, full.Pgg_over), ('qqbar', full.Pgq, full.Pgq_over)])
def test_kernels_and_overestimates(channel, kernel, over):
    splitting = lambda z: kernel(z, 100., .935, .1)
    for z in np.linspace(.00001, .99999, 301):
        assert 0 < splitting(z) <= over(z)
        assert full.inversetGamma(full.tGamma(z, .1, channel), .1, channel) == pytest.approx(z)
    if channel == 'gg':
        # Herwig's full-z convention includes 1/2 for identical daughters.
        for z in (.1, .5, .9):
            assert splitting(z) == pytest.approx(full.CA*(z/(1-z) + (1-z)/z + z*(1-z)))
    elif channel == 'qqbar':
        assert quad(splitting, 0, 1)[0] == pytest.approx(2*full.TR/3)


def test_competing_gluon_channels_against_integrated_sudakov(monkeypatch):
    # Integrate the physical kernels independently with a fixed coupling.
    # This checks the factor from d(qtilde^2)/qtilde^2, no-emission probability,
    # highest-scale distribution and the sum over quark flavours together.
    coupling, qmax, ptmin, nf = .2, 30., .9, 5
    random.seed(8301)
    over = coupling/(2*math.pi)
    monkeypatch.setattr(full, 'alphaS', lambda *args: over)

    def rates(x):
        lo = .5*(1-math.sqrt(max(0., 1-4*ptmin/math.exp(x))))
        hi = 1-lo
        gg = lambda z: full.CA*(math.log(z)-math.log1p(-z)-2*z+z*z/2-z**3/3)
        qq = lambda z: full.TR*(z-z*z+2*z**3/3)
        return np.array([gg(hi)-gg(lo), nf*(qq(hi)-qq(lo))])*coupling/math.pi

    xmin, xmax = math.log(4*ptmin), math.log(qmax)

    def survival(x):
        return math.exp(-quad(lambda u: sum(rates(u)), x, xmax, epsabs=1.e-9)[0])

    expected_qq = quad(lambda x: rates(x)[1]*survival(x), xmin, xmax)[0]
    n = 16000
    candidates = [full.Choose_Emission(21, qmax, ptmin, over) for _ in range(n)]

    def check_frequency(observed, probability):
        assert abs(observed/n-probability) < 5*math.sqrt(probability*(1-probability)/n)+2/n

    for q in (4*ptmin, 8., 16.):
        check_frequency(sum(c is None or c[0] <= q*q for c in candidates), survival(math.log(q)))
    check_frequency(sum(c is not None and c[5] == 'qqbar' for c in candidates), expected_qq)
    for flavour in range(1, nf+1):
        check_frequency(sum(c is not None and c[6] == flavour for c in candidates), expected_qq/nf)


def test_winning_channel_and_no_emission():
    def emission(channel, Q, flavour=0):
        return [Q*Q, .5, Q/4, Q*Q/4, 0., channel, flavour]
    with patch.object(full, 'Next_Emission', side_effect=[
            emission('gg', 9.), None, emission('qqbar', 20., 2),
            None, emission('qqbar', 12., 4), None]):
        assert full.Choose_Emission(21, 100., .9, .1)[6] == 2
    with patch.object(full, 'Next_Emission', return_value=None):
        assert full.Choose_Emission(21, 100., .9, .1) is None
    for channel in ('qg', 'gg', 'qqbar'):
        assert full.Next_Emission(3.6, .9, .1, channel) is None


def check_event(particles):
    final = [p for p in particles if p[1] == 1]
    momenta = np.array([p[2:6] for p in final])
    incoming = np.sum([p[2:6] for p in particles if p[1] == -1], axis=0)
    assert np.all(np.isfinite(momenta)) and np.all(momenta[:, 3] > 0)
    np.testing.assert_allclose(momenta.sum(axis=0), incoming, atol=1.e-8, rtol=0.)
    np.testing.assert_allclose(momenta[:, 3]**2 - np.sum(momenta[:, :3]**2, axis=1), 0., atol=1.e-8)
    colors = Counter(p[7] for p in final if p[7])
    assert colors == Counter(p[8] for p in final if p[8])
    assert all(n == 1 for n in colors.values())
    assert all(p[7] != p[8] for p in final if p[0] == 21)
    assert all(sum((p[0] == f)-(p[0] == -f) for p in final) == 0 for f in range(1, 6))


def test_both_daughters_radiate_and_tree_conserves_momentum_and_flavour():
    random.seed(90210)
    over = full.get_alphaS_over(.935)
    channels = Counter()
    secondary_quark_radiates = False
    gluon_children_radiate = [False, False]
    with LHEFile(INPUT) as source:
        for event in islice(source, 500):
            result, _ = reconstruct_event(full.Shower, event.particles, .9, over, assign_colors=False)
            check_event(result)
            for parent in (p for p in event.particles if p[1] == 1):
                partner, Q2start = full.find_color_partner(parent, event.particles)
                tree = full.EvolveParticle(parent, .9, Q2start, over)
                leaves = full.reconstructSudakov(parent, partner, tree)
                momenta = np.array([p[2:6] for p in leaves])
                # Light-cone and transverse momentum conservation before recoil.
                assert sum(momenta[:, 3]+momenta[:, 2]) == pytest.approx(2*np.linalg.norm(parent[2:5]))
                np.testing.assert_allclose(momenta[:, :2].sum(axis=0), 0., atol=1.e-12)
                for part in full.shower_partons(tree):
                    b = part['Emission']
                    if b is None:
                        continue
                    channels[b[5]] += 1
                    assert b[0] < part['Q2start']
                    assert b[2] >= .9
                    for child, fraction in zip(part['children'], (b[1], 1-b[1])):
                        assert child['Q2start'] == pytest.approx(fraction**2*b[0])
                    if b[5] == 'qqbar':
                        secondary_quark_radiates |= any(c['Emission'] is not None for c in part['children'])
                    if b[5] == 'gg':
                        gluon_children_radiate = [a or c['Emission'] is not None
                                                  for a, c in zip(gluon_children_radiate, part['children'])]
    assert all(channels[c] > 0 for c in ('qg', 'gg', 'qqbar'))
    assert secondary_quark_radiates and all(gluon_children_radiate)


def test_no_fixed_limit_on_number_of_particles():
    def split(pid, Q, Qcut, over):
        if Q > 10.:
            return [Q*Q/4, .5, Q/8, Q*Q/16, 0., 'gg', 0]
        return None
    with patch.object(full, 'Choose_Emission', side_effect=split):
        root = full.EvolveParticle([21, 1, 0, 0, 0, 0, 0, 501, 502], .9, 1.e14, .1)
    leaves = [p for p in full.shower_partons(root) if not p['children']]
    assert len(leaves) > 100
    assert all(p['Q2start'] <= 100. for p in leaves)


def test_reconstruction_failure_keeps_the_hard_event_and_statistics():
    with LHEFile(INPUT) as source:
        event = next(iter(source))
    original = shower_cli.GlobalMomCons
    calls, snapshots, statistics = [], [], {}
    def retry_once(particles, jets):
        calls.append([p for p in particles if p[1] == -1])
        snapshots.append(dict(statistics))
        if len(calls) == 1:
            raise KinematicsError('Synthetic impossible shower')
        return original(particles, jets)
    with patch.object(shower_cli, 'GlobalMomCons', side_effect=retry_once):
        result, retries = reconstruct_event(full.Shower, event.particles, .9,
                  full.get_alphaS_over(.935), assign_colors=False, statistics=statistics)
    assert retries == 1 and calls[0] == calls[1]
    assert statistics == snapshots[-1]  # Rejected attempts do not inflate counts.
    check_event(result)


def test_quark_only_mode_matches_the_original_shower_event_by_event(monkeypatch):
    # This is the direct bridge back to the original tutorial: the same random
    # numbers must give the same quark-line shower, momenta and colour tags.
    monkeypatch.setattr(full, 'gluon_splittings', False)
    over = full.get_alphaS_over(.935)
    with LHEFile(INPUT) as source:
        for index, event in enumerate(islice(source, 100)):
            random.seed(index)
            expected, tries_a = reconstruct_event(quark_only.Shower, event.particles, .9, over)
            random.seed(index)
            actual, tries_b = reconstruct_event(full.Shower, event.particles, .9, over, assign_colors=False)
            assert tries_a == tries_b
            np.testing.assert_allclose(actual, expected, rtol=1.e-10, atol=1.e-9)


def test_cli_seed_counts_and_weight_preservation(tmp_path):
    source_path, first, second = (tmp_path/name for name in ('hard.lhe', 'a.lhe', 'b.lhe'))
    with LHEFile(INPUT) as source, lhe_output(source_path, source.preamble) as out:
        for event in islice(source, 3):
            event.header[2] = '-2.5D+00'
            event.extra = '<rwgt><wgt id="scale">-3.0</wgt></rwgt>\n'
            write_event(out, event.particles, event)
    with contextlib.redirect_stdout(StringIO()):
        full.main([str(source_path), '-o', str(first), '-n', '3', '--seed', '41'])
        full.main(['--seed', '41', '-n', '3', '-o', str(second), str(source_path)])
    assert first.read_bytes() == second.read_bytes()
    with LHEFile(first) as source:
        events = list(source)
    assert len(events) == 3
    for event in events:
        assert event.weight == -2.5 and event.multiweights == {'scale': -3.}
        check_event(event.particles)
    with contextlib.redirect_stdout(StringIO()):
        full.main([str(source_path), '-o', str(first), '-n', '0'])
    with LHEFile(first) as source:
        assert list(source) == []
