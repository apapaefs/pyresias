"""Numerical regression checks against the saved Herwig comparison coupling.

Run from the repository root: python -m unittest discover -s tests -v
"""
import importlib
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from alphaS_HW import alphaS
import pyresias_qtilde

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/herwig-7.3.0-coupling.json").read_text()
)


def herwig_alpha(scale):
    """Evaluate the independent saved Lambdas with Herwig's coefficient convention."""
    scale = max(scale, REFERENCE["freeze_scale_GeV"])
    nf = 3 + sum(scale >= threshold for threshold in REFERENCE["thresholds_GeV"])
    lam = REFERENCE["lambdas_GeV_nf3_to_nf6"][nf - 3]
    log_scale = math.log((scale / lam)**2)
    b0, b1 = 11. - 2.*nf/3., 51. - 19.*nf/3.
    return 4.*math.pi/(b0*log_scale) * (1. - 2.*b1/b0**2 * math.log(log_scale)/log_scale)


class RunningCouplingTests(unittest.TestCase):
    def test_input_coupling_is_reproduced(self):
        for order in (1, 2):
            for value in (.1074, .118, .126):
                with self.subTest(order=order, value=value):
                    coupling = alphaS(value, 91.1876, order=order)
                    self.assertAlmostEqual(coupling.alphasQ(91.1876), value, delta=2.e-14)

    def test_matches_saved_herwig_lambdas_and_curve(self):
        coupling = alphaS(.1074, 91.1876)
        np.testing.assert_allclose(
            [coupling.lambdas[nf] for nf in range(3, 7)],
            REFERENCE["lambdas_GeV_nf3_to_nf6"], rtol=1.e-10, atol=0.,
        )
        scales = np.geomspace(.935, 1.e6, 2001)
        np.testing.assert_allclose(
            [coupling.alphasQ(q) for q in scales],
            [herwig_alpha(q) for q in scales], rtol=1.e-10, atol=0.,
        )

    def test_thresholds_are_continuous_with_custom_thresholds_too(self):
        for order in (1, 2):
            for mc, mb, mt in ((1.6, 5., 172.69), (1.25, 4.2, 174.2)):
                coupling = alphaS(.1074, 91.1876, mc=mc, mb=mb, mt=mt, order=order)
                for low_nf, threshold in zip((3, 4, 5), (mc, mb, mt)):
                    with self.subTest(order=order, threshold=threshold):
                        below = coupling.as_from_lambda(threshold, coupling.lambdas[low_nf], low_nf)
                        above = coupling.as_from_lambda(threshold, coupling.lambdas[low_nf + 1], low_nf + 1)
                        self.assertAlmostEqual(below, above, delta=2.e-14)
                        self.assertAlmostEqual(
                            coupling.alphasQ(np.nextafter(threshold, 0.)),
                            coupling.alphasQ(np.nextafter(threshold, math.inf)), delta=2.e-14,
                        )

    def test_invalid_parameters_and_landau_pole_are_rejected(self):
        for options in ({"order": 3}, {"mc": 6.}, {"mt": 80.}, {"mb": math.nan}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                alphaS(.1074, 91.1876, **options)
        coupling = alphaS(.1074, 91.1876)
        for scale in (0., -1., math.nan, math.inf, coupling.lambdas[3]):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                coupling.alphasQ(scale)


class ShowerCouplingTests(unittest.TestCase):
    def setUp(self):
        self.shower = vars(importlib.reload(pyresias_qtilde))
        self.addCleanup(importlib.reload, pyresias_qtilde)

    def test_frozen_curve_matches_herwig_and_respects_veto_bound(self):
        shower = self.shower
        over = shower["get_alphaS_over"](shower["Qc"])
        self.assertAlmostEqual(2.*math.pi*over, REFERENCE["alpha_s_at_freeze"], delta=2.e-12)
        for scale in np.concatenate(([0., .9, .91, .935], np.geomspace(1.e-8, 1.e6, 2001))):
            value = shower["alphaS"]((4.*scale)**2, .5, shower["Qc"], over)
            self.assertAlmostEqual(2.*math.pi*value, herwig_alpha(scale), delta=2.e-12)
            self.assertGreaterEqual(value / over, 0.)
            self.assertLessEqual(value / over, 1.)

    def candidate_at_pt(self, pt, emission_cutoff, over):
        # Force a z=1/2 candidate at a chosen pT through the real proposal code.
        shower = self.shower
        start = 10.
        t = (4.*pt)**2
        lower = shower["tGamma"](shower["zm_over"](start**2, emission_cutoff), over)
        upper = shower["tGamma"](shower["zp_over"](start**2, emission_cutoff), over)
        draws = iter(((t/start**2)**(upper-lower),
                      (shower["tGamma"](.5, over)-lower)/(upper-lower),
                      0., 1.-1.e-9))
        shower["random"] = lambda: next(draws)
        return shower["Generate_Emission"](start, emission_cutoff, over)

    def test_emission_cutoff_and_coupling_freeze_are_independent(self):
        over = self.shower["get_alphaS_over"](.935)
        allowed = self.candidate_at_pt(.91, .9, over)
        self.assertTrue(allowed[4])
        self.assertAlmostEqual(allowed[2], .91**2, delta=1.e-13)
        below_cutoff = self.candidate_at_pt(.91, .92, over)
        self.assertFalse(below_cutoff[4])

    def test_invalid_overestimate_fails_instead_of_silently_accepting(self):
        over = self.shower["get_alphaS_over"](.935) * .5
        with self.assertRaisesRegex(RuntimeError, "Invalid alpha_s veto probability"):
            self.candidate_at_pt(.91, .9, over)

    def test_coupling_bound_for_alternative_freeze_scales_and_cmw_options(self):
        # The alternative CMW conventions are checked for a valid coupling
        # bound here; the saved Herwig reference uses no CMW conversion.
        for scheme in ("None", "Linear", "Factor"):
            self.shower["CMW"] = scheme
            for freeze in (.8, .935, 2.):
                over = self.shower["get_alphaS_over"](freeze)
                values = [self.shower["alphaS_at_scale"](q, freeze)
                          for q in np.geomspace(1.e-8, 1.e4, 401)]
                with self.subTest(scheme=scheme, freeze=freeze):
                    self.assertTrue(all(0. < value <= over for value in values))
                    self.assertLess(values[-1], values[0])


if __name__ == "__main__":
    unittest.main()
