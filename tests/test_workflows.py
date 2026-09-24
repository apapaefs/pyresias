"""Regression tests for the actual tutorial entry points and physical invariants."""
from collections import Counter
import contextlib
from io import StringIO
from itertools import islice
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from alphaS import alphaS
from kinematics import GetRotationMatrixAB, boost, GlobalMomCons, KinematicsError
from lhe_io import LHEFile, write_event, lhe_output
from shower_cli import reconstruct_event
import pyresias
import pyresias_qtilde as qtilde
import pyresias_test as tutorial
from analysis.lhe_analyzer import Histogram, event_observables

INPUT = REPO / 'data/eejj_ECM206.lhe.gz'


def hard_event():
    with LHEFile(INPUT) as source:
        return next(iter(source))


def check_event(case, particles):
    final = np.asarray([p[2:6] for p in particles if p[1] == 1])
    incoming = np.sum([p[2:6] for p in particles if p[1] == -1], axis=0)
    case.assertTrue(np.all(final[:, 3] > 0.))
    case.assertTrue(np.all(np.isfinite(final)))
    np.testing.assert_allclose(final.sum(axis=0), incoming, atol=1.e-7, rtol=0.)
    np.testing.assert_allclose(final[:, 3]**2 - np.sum(final[:, :3]**2, axis=1),
                               0., atol=1.e-7, rtol=0.)
    colors = Counter(int(p[7]) for p in particles if p[1] == 1 and p[7])
    anticolors = Counter(int(p[8]) for p in particles if p[1] == 1 and p[8])
    case.assertEqual(colors, anticolors)
    case.assertTrue(all(n == 1 for n in colors.values()))
    for p in particles:
        if p[0] == 21 and p[1] == 1:
            case.assertNotEqual(p[7], p[8])


class KinematicsTests(unittest.TestCase):
    def test_parallel_antiparallel_and_general_rotations(self):
        for a, b in (([0, 0, 1], [0, 0, -4]), ([1, 2, 3], [2, 4, 6]),
                     ([1, 0, 0], [0, 1, 0]), ([1, 2, 3], [-2, 4, 1])):
            matrix = GetRotationMatrixAB(a, b)
            np.testing.assert_allclose(matrix @ (np.array(a)/np.linalg.norm(a)),
                                       np.array(b)/np.linalg.norm(b), atol=1.e-14)
            np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1.e-14)
            self.assertAlmostEqual(np.linalg.det(matrix), 1.)
        with self.assertRaises(ValueError):
            GetRotationMatrixAB([0, 0, 0], [1, 0, 0])

    def test_axis_aligned_and_small_boosts_preserve_mass(self):
        p = np.array([3., 4., 5., 10.])
        for beta in ([0., 0., .7], [.2, 0., 0.], [0., 0., 0.], [1.e-10, 0., 0.], [-.8, .1, 0.]):
            result = boost(p, beta)
            np.testing.assert_allclose(boost(result, -np.array(beta)), p, atol=1.e-13)
            self.assertAlmostEqual(result[3]**2 - np.dot(result[:3], result[:3]), 50., places=11)
        with self.assertRaises(KinematicsError):
            boost(p, [0, 0, 1])

    def test_no_emission_beam_aligned_event(self):
        particles = [[11, -1, 0., 0., 103., 103., 0., 0, 0],
                     [-11, -1, 0., 0., -103., 103., 0., 0, 0],
                     [1, 1, 0., 0., 103., 103., 0., 501, 0],
                     [-1, 1, 0., 0., -103., 103., 0., 0, 501]]
        for module in (pyresias, qtilde):
            result, _ = reconstruct_event(module.Shower, particles, 500., module.get_alphaS_over(.935))
            check_event(self, result)
            np.testing.assert_allclose(result, particles, atol=1.e-12)

    def test_mass_above_available_energy_is_rejected_and_retried(self):
        particles = hard_event().particles
        parents = [p for p in particles if p[1] == 1]
        bad_jets = [[p, [[p[0], 1, 0., 0., 0., 150., 150.]]] for p in parents]
        with self.assertRaises(KinematicsError):
            GlobalMomCons(particles, bad_jets)
        valid = [[p, [list(p)]] for p in parents]
        attempts = iter([(particles, bad_jets), (particles, valid)])
        result, retries = reconstruct_event(lambda *args: next(attempts), particles, .9, .1)
        self.assertEqual(retries, 1)
        check_event(self, result)

    def test_seeded_showers_conserve_energy_momentum_and_color(self):
        for module in (pyresias, qtilde):
            random.seed(42)
            cutoff = .9 if module is qtilde else .935
            with LHEFile(INPUT) as source:
                for event in islice(source, 200):
                    result, _ = reconstruct_event(module.Shower, event.particles, cutoff,
                                                      module.get_alphaS_over(.935))
                    check_event(self, result)


class FileAndCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def test_event_limit_seed_and_argument_order(self):
        for module in (pyresias, qtilde):
            first, second = self.path/'a.lhe', self.path/'b.lhe'
            with contextlib.redirect_stdout(StringIO()):
                module.main(['-n', '3', '--seed', '13', '-o', str(first), str(INPUT)])
                module.main([str(INPUT), '-o', str(second), '--seed', '13', '-n', '3'])
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with LHEFile(first) as source:
                self.assertEqual(len(list(source)), 3)
            with contextlib.redirect_stdout(StringIO()):
                module.main([str(INPUT), '-n', '0', '-o', str(first)])
            with LHEFile(first) as source:
                self.assertEqual(list(source), [])

    def test_weights_metadata_reweights_and_fortran_numbers(self):
        event = hard_event()
        event.header[1:6] = ['4242', '-2.5D+00', '70.', '.0073', '.119']
        event.extra = '<rwgt><wgt id="scale_up"> -3.0D+00 </wgt></rwgt>\n'
        modified, output = self.path/'input.lhe', self.path/'output.lhe'
        with LHEFile(INPUT) as source, lhe_output(modified, source.preamble) as out:
            write_event(out, event.particles, event)
        with contextlib.redirect_stdout(StringIO()):
            qtilde.main([str(modified), '-o', str(output)])
        with LHEFile(output) as source:
            self.assertIn('<init>', source.preamble)
            records = list(source)
        self.assertEqual(records[0].header[1:], event.header[1:])
        self.assertEqual(records[0].weight, -2.5)
        self.assertEqual(records[0].multiweights, {'scale_up': -3.})

    def test_bad_input_and_failed_run_do_not_overwrite_output(self):
        event = hard_event()
        event.particles[-1][0] = 21
        modified, output = self.path/'input.lhe', self.path/'output.lhe'
        with LHEFile(INPUT) as source, lhe_output(modified, source.preamble) as out:
            write_event(out, event.particles, event)
        output.write_text('keep existing output')
        with contextlib.redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            qtilde.main([str(modified), '-o', str(output)])
        self.assertEqual(output.read_text(), 'keep existing output')
        self.assertEqual(list(self.path.glob('*.part')), [])

    def test_reader_rejects_truncation_and_lfs_pointer(self):
        for text in ('version https://git-lfs.github.com/spec/v1\n',):
            path = self.path/'bad.lhe'
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, 'LFS'):
                with LHEFile(path):
                    pass
        with LHEFile(INPUT) as source:
            preamble = source.preamble
        path.write_text(preamble + '<event>\n4 1 1 1 1 1\n')
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            with LHEFile(path) as source:
                list(source)

    def test_imports_are_quiet_and_do_not_run_cli(self):
        result = subprocess.run([sys.executable, '-c',
            'import pyresias, pyresias_qtilde, pyresias_test; import analysis.lhe_analyzer'],
            cwd=REPO, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, '')
        self.assertEqual(result.stderr, '')

    def test_hepmc_roundtrip(self):
        import pyhepmc
        from HEPMCWriter import WriteHepMC
        output = self.path/'events.hepmc'
        with pyhepmc.open(str(output), 'w') as writer:
            WriteHepMC(writer, [hard_event().particles], weights=[-2.5], cross_section=(12., .3))
        with pyhepmc.open(str(output)) as reader:
            event = reader.read()
        self.assertEqual(event.weights, [-2.5])
        self.assertEqual(len(event.particles), 4)
        self.assertAlmostEqual(event.cross_section.xsec(), 12.)

    def test_legacy_lhe_writer_roundtrip(self):
        from LHEWriter import init_lhe, write_lhe, finalize_lhe
        output = self.path/'legacy.lhe'
        stream = init_lhe(output, 12., .3, 206.)
        write_lhe(stream, [hard_event().particles], 206.**2, weights=[-2.5])
        finalize_lhe(stream)
        with LHEFile(output) as source:
            events = list(source)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].weight, -2.5)
        check_event(self, events[0].particles)

    def test_tutorial_empty_and_below_cutoff_plots(self):
        for n in (0, 5):
            output = self.path/f'tutorial plots {n}'
            result = subprocess.run([sys.executable, str(REPO/'pyresias_test.py'),
                                     '-n', str(n), '-Q', '1', '-c', '1', '-o', str(output)],
                                    capture_output=True, text=True, check=True)
            self.assertIn('0 emissions', result.stdout)
            self.assertEqual(len(list(output.glob('*.pdf'))), 5)

    def test_tutorial_running_coupling_with_both_samplers(self):
        for method in ('Overestimated', 'Numerical'):
            result = subprocess.run([sys.executable, str(REPO/'pyresias_test.py'),
                                     '-n', '25', '-Q', '100', '--coupling', 'pt',
                                     '--method', method, '--no-plots'],
                                    capture_output=True, text=True, check=True)
            self.assertIn('Evolved 25 branches', result.stdout)
            self.assertNotIn('; 0 emissions', result.stdout)

    def test_analysis_one_two_and_three_samples_including_empty_gluons(self):
        from analysis.lhe_analyzer import main
        path = self.path/'input.lhe'
        with LHEFile(INPUT) as source, lhe_output(path, source.preamble) as out:
            write_event(out, hard_event().particles, hard_event())
        for count in (1, 2, 3):
            output = self.path/str(count)
            with contextlib.redirect_stdout(StringIO()):
                main([*[str(path)]*count, '-o', str(output), '--observables', 'ptg', 'ng'])
            self.assertEqual(len(list(output.glob('*.pdf'))), 2)
            data = json.loads((output/'histograms.json').read_text())
            self.assertFalse(data['samples'][0]['histograms']['ptg']['normalization_defined'])


class AnalysisAndTutorialTests(unittest.TestCase):
    def test_histogram_uses_bin_width_and_event_covariance(self):
        hist = Histogram([0., 1., 3.])
        hist.add([.5, .6])
        hist.add([2.])
        result = hist.result()
        np.testing.assert_allclose(result['density'], [2/3, 1/6])
        np.testing.assert_allclose(result['standard_error'], [4/9, 2/9])
        self.assertAlmostEqual(np.dot(result['density'], [1., 2.]), 1.)

    def test_analysis_final_state_and_fastjet_fourvector_order(self):
        event = hard_event().particles
        event.append([21, 2, 1., 2., 3., 10., 0., 501, 502])
        data = event_observables(event, jets=True)
        self.assertEqual(data['ng'], [0])
        self.assertEqual(data['nq'], [2])
        self.assertEqual(data['njets'], [2])
        pt = np.hypot(event[2][2], event[2][3])
        np.testing.assert_allclose(data['ptj'], [pt, pt], rtol=1.e-9)

    def test_legacy_coupling_orders_and_thresholds(self):
        for order in (1, 2):
            coupling = alphaS(.118, 91.1876, order=order)
            self.assertAlmostEqual(coupling.alphasQ(91.1876), .118)
            for q in (1.25, 4.2):
                self.assertAlmostEqual(coupling.alphasQ(np.nextafter(q, 0.)),
                                       coupling.alphasQ(q), places=12)

    def test_proposal_cutoff_and_matching_z_interval(self):
        for sampler in (tutorial.Get_tEmission, tutorial.Get_tEmission_direct):
            result = sampler(2., 1., .5, .1)
            self.assertEqual(len(result), 4)
            self.assertFalse(result[2])
        tutorial.tMethod = 'Overestimated'
        original = tutorial.Get_zEmission
        scales = []
        def record_scale(t, *args):
            scales.append(t)
            return original(t, *args)
        with patch.object(tutorial, 'Get_zEmission', side_effect=record_scale), \
             patch.object(tutorial, 'random', side_effect=[.9, .5, 0., 0.]):
            tutorial.Generate_Emission(100., 1., tutorial.get_alphaS_over(100., 1.))
        self.assertEqual(scales, [10000.])

    def test_numerical_and_analytic_proposals_agree_statistically(self):
        means = []
        for method in ('Overestimated', 'Numerical'):
            random.seed(231)
            tutorial.tMethod = method
            emissions = [len(tutorial.Evolve(30., 1., tutorial.get_alphaS_over(30., 1.)))
                         for _ in range(1000)]
            means.append((np.mean(emissions), np.var(emissions, ddof=1)/len(emissions)))
        self.assertLess(abs(means[0][0]-means[1][0]), 5*math.sqrt(means[0][1]+means[1][1]))
        tutorial.tMethod = 'Overestimated'

    def test_fixed_support_splitting_kernel(self):
        from scipy.integrate import quad
        random.seed(2345)
        samples = tutorial.sample_splitting(20000)
        expected = quad(lambda z: z*tutorial.Pqq(z), .01, .99)[0] / quad(tutorial.Pqq, .01, .99)[0]
        self.assertLess(abs(samples.mean()-expected), 5*samples.std()/math.sqrt(len(samples)))


if __name__ == '__main__':
    unittest.main()
