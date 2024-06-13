import numpy as np

# QCD Constants:
# QCD quark charge = (N^2 - 1)/2N for N colours
NC = 3
TR = 1/2
CA = NC
CF = (NC * NC - 1.) / (2. * NC)

# the alphaS class
class alphaS:
    """An alphaS class"""
    def __init__(self, asmz, mz, mb=4.75, mc=1.27, order=1):
        self.order = order
        self.asmz = asmz
        self.mz = mz
        self.mz2 = mz**2
        self.mb2 = mb**2
        self.mc2 = mc**2
        self.asmb = self.alphasQ(mb)
        self.asmc = self.alphasQ(mc)
        
    # NLO alphaS:
    def As1(self, t):
        if t >= self.mb2:
            tref = self.mz2
            asref = self.asmz
            b0 = self.Beta0(5) / (2. * np.pi)
            b1 = self.Beta1(5) / (2. * np.pi)**2
        elif t >= self.mc2:
            tref = self.mb2
            asref = self.asmb
            b0 = self.Beta0(4) / (2. * np.pi)
            b1 = self.Beta1(4) / (2. * np.pi)**2
        else:
            tref = self.mc2
            asref = self.asmc
            b0 = self.Beta0(3) / (2. * np.pi)
            b1 = self.Beta1(3) / (2. * np.pi)**2
        w = 1. + b0 * asref * np.log(t / tref)
        return asref / w * (1. - b1 / b0 * asref * np.log(w) / w)

    # LO alphaS
    def As0(self, t):
        if t >= self.mb2:
            tref = self.mz2
            asref = self.asmz
            b0 = self.Beta0(5) / (2. * np.pi)
        elif t >= mc2:
            tref = self.mb2
            asref = self.asmb
            b0 = self.Beta0(4) / (2. * np.pi)
        else:
            tref = self.mc2
            asref = self.asmc
            b0 = self.Beta0(3) / (2. * np.pi)
        return 1. / (1. / asref + b0 * log(t / tref))

    # beta functions:
    def Beta0(self,nf):
        return (11. / 6. * CA) - (2. / 3. * TR * nf)

    def Beta1(self,nf):
        return (17. / 6. * CA**2) - ((5. / 3. * CA + CF) * TR * nf)

    # function to access alphaS at scale Q (not squared!)
    def alphasQ(self, Q):
        if self.order == 0:
            return self.As1(Q**2)
        else:
            return self.As1(Q**2)







