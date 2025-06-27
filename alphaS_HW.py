import numpy as np

NC = 3
TR = 1/2
CA = NC
CF = (NC * NC - 1.) / (2. * NC)

class alphaS:
    """An alphaS class with exact threshold matching (Herwig/QCD style)"""
    def __init__(self, asmz, mz, mb=4.2, mc=1.25, mt=174.2, order=2):
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
        b0 = self.Beta0(nf) / (2. * np.pi)
        b1 = self.Beta1(nf) / (2. * np.pi)**2
        t = 1. / (b0 * asQ)
        if self.order == 1:
            return Q * np.exp(-t / 2.)
        elif self.order == 2:
            return Q * np.exp(-t / 2.) * (b0 * asQ) ** (-b1 / (2. * b0**2))
        else:
            raise ValueError("Order must be 1 or 2")

    def as_from_lambda(self, Q, Lambda, nf):
        b0 = self.Beta0(nf) / (2. * np.pi)
        b1 = self.Beta1(nf) / (2. * np.pi)**2
        t = np.log(Q**2 / Lambda**2)
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
