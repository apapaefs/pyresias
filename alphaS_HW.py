import numpy as np
from scipy.optimize import brentq

NC = 3
TR = 1/2
CA = NC
CF = (NC * NC - 1.) / (2. * NC)

class alphaS:
    """Herwig's one/two-loop parametrization with numerical threshold matching.

    The default thresholds match the Herwig 7.3 comparison configuration.
    They set the flavour number in the coupling, not shower kinematic masses.
    Freezing at low scales is handled by the shower.
    """
    def __init__(self, asmz, mz, mb=5.0, mc=1.6, mt=172.69, order=2):
        if order not in (1, 2):
            raise ValueError("Order must be 1 or 2")
        if not all(np.isfinite(x) and x > 0 for x in (asmz, mz, mc, mb, mt)):
            raise ValueError("The coupling, reference scale and thresholds must be positive and finite")
        if not mc < mb <= mz < mt:
            raise ValueError("Expected mc < mb <= mz < mt for the five-flavour input coupling")
        self.order = order
        self.asmz = asmz
        self.mz = mz
        self.mb = mb
        self.mc = mc
        self.mt = mt
        # Compute LambdaQCD for each nf region
        self.lambdas = self.compute_lambdas()

    def Beta0(self, nf):
        return (11. / 6. * CA) - (2. / 3. * TR * nf)

    def Beta1(self, nf):
        return (17. / 6. * CA**2) - ((5. / 3. * CA + CF) * TR * nf)

    def lambda_from_as(self, Q, asQ, nf):
        """Invert the same running formula used when evaluating alpha_s.

        The usual asymptotic expression for Lambda is not the inverse of the
        truncated two-loop formula. Solve instead for t = log(Q^2/Lambda^2),
        which keeps the numerical search above the Landau pole.
        """
        if not all(np.isfinite(x) and x > 0 for x in (Q, asQ)):
            raise ValueError("Q and alpha_s(Q) must be positive and finite")
        if nf not in (3, 4, 5, 6):
            raise ValueError("The coupling supports three to six active flavours")
        b0 = self.Beta0(nf) / (2. * np.pi)
        t = 1. / (b0 * asQ)
        if self.order == 2:
            # For nf=3..6 the positive root lies between 1 and the one-loop t.
            t = brentq(lambda value: self._as_from_log(value, nf) - asQ,
                       min(1., t), max(1., t), xtol=1.e-14, rtol=1.e-14)
        return Q * np.exp(-t / 2.)

    def as_from_lambda(self, Q, Lambda, nf):
        if not (np.isfinite(Q) and np.isfinite(Lambda) and Q > Lambda > 0):
            raise ValueError("Running alpha_s requires finite Q > Lambda > 0")
        return self._as_from_log(2. * np.log(Q / Lambda), nf)

    def _as_from_log(self, t, nf):
        b0 = self.Beta0(nf) / (2. * np.pi)
        b1 = self.Beta1(nf) / (2. * np.pi)**2
        if self.order == 1:
            return 1. / (b0 * t)
        elif self.order == 2:
            return 1. / (b0 * t) * (1. - b1 * np.log(t) / (b0**2 * t))
        else:
            raise ValueError("Order must be 1 or 2")

    def compute_lambdas(self):
        # Start with nf=5 at MZ
        lambdas = {}
        # Step 1: Lambda_5 from MZ
        Lambda5 = self.lambda_from_as(self.mz, self.asmz, 5)
        lambdas[5] = Lambda5
        # Step 2: Match down to nf=4 at mb
        as_mb_5 = self.as_from_lambda(self.mb, Lambda5, 5)
        Lambda4 = self.lambda_from_as(self.mb, as_mb_5, 4)
        lambdas[4] = Lambda4
        # Step 3: Match down to nf=3 at mc
        as_mc_4 = self.as_from_lambda(self.mc, Lambda4, 4)
        Lambda3 = self.lambda_from_as(self.mc, as_mc_4, 3)
        lambdas[3] = Lambda3
        # Step 4: Match up to nf=6 at mt (if needed)
        as_mt_5 = self.as_from_lambda(self.mt, Lambda5, 5)
        Lambda6 = self.lambda_from_as(self.mt, as_mt_5, 6)
        lambdas[6] = Lambda6
        return lambdas

    def alphasQ(self, Q):
        # Choose nf and Lambda for the scale
        if Q < self.mc:
            nf = 3
        elif Q < self.mb:
            nf = 4
        elif Q < self.mt:
            nf = 5
        else:
            nf = 6
        Lambda = self.lambdas[nf]
        return self.as_from_lambda(Q, Lambda, nf)

# Example usage:
# asrun = alphaS(0.118, 91.1876)
# print(asrun.alphasQ(10.0))  # αs at 10 GeV
# print(asrun.alphasQ(1.0))   # αs at 1 GeV
