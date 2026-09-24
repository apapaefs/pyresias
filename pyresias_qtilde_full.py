"""Massless angular-ordered shower with q->qg, g->gg and g->qqbar.

Read alongside pyresias_qtilde.py: the helper names and order are the same.
The additions are gluon kernels, channel competition and evolution of both
daughters. See docs/full-shower.md for the Herwig conventions and comparison.
"""
from random import random
from itertools import count
import math
import sys
import numpy as np
from prettytable import PrettyTable
from alphaS_HW import alphaS, CF, CA, TR
from shower_cli import run_shower
from kinematics import RotateMomentaLab, dot4vec, KinematicsError

debug = False
Qc = .935
pTmin = .900
Nflavours = 5
gluon_splittings = True

# function to print the emission information once they have been genrated:
def PrintEmissions(EmissionsArray):
    tbl = PrettyTable(["#", "Evo scale [GeV]", '1-z', 'pT [GeV]', 'virt. mass in a->bc [GeV]'])
    for i in range(len(EmissionsArray)):
        tbl.add_row([i, np.sqrt(EmissionsArray[i][0]), 1-EmissionsArray[i][1], EmissionsArray[i][2], np.sqrt(EmissionsArray[i][3])])
    print(tbl)

# function to print the momenta information once they have been genrated:
def PrintMomenta(MomentaArray):
    tbl = PrettyTable(["n", "id", "status", 'px [GeV]', 'py [GeV]', 'pz [GeV]', 'E [GeV]', 'm [GeV]'])
    for i in range(len(MomentaArray)):
        tbl.add_row([i, MomentaArray[i][0], MomentaArray[i][1], MomentaArray[i][2], MomentaArray[i][3], MomentaArray[i][4], MomentaArray[i][5], MomentaArray[i][6]])
    print(tbl)


# initialize alphaS class: pass the value of alphaS at mz, and mz
aS = alphaS(0.1074, 91.1876, mc=1.6, mb=5.0, mt=172.69, order=2)

# CMW scheme:
CMW = 'None' # 'Linear' or 'Factor' or 'None'

#################################################

# the Kg function for the CMW Scheme
def Kg():
    Nf=5 # 5-flavors
    return 3.*(67./18.-1./6.*np.pi**2)-5./9.*Nf

# the q -> q + g splitting function
def Pqq(z, t, Qfreeze, aSover):
    return CF * (1. + z*z) / (1. - z)

# the q -> q + g splitting function *overestimate*
def Pqq_over(z): return 2.*CF/(1.-z)

# ADDED: gluon kernels, in Herwig's full-z convention.
# Pgg includes the factor 1/2 for identical gluon daughters.
def Pgg(z, t, Qfreeze, aSover):
    return CA * (1. - z*(1. - z))**2 / (z*(1. - z))


def Pgg_over(z): return CA/(z*(1.-z))


# One g -> q qbar channel per massless quark flavour; z belongs to the quark.
def Pgq(z, t, Qfreeze, aSover):
    return TR * (z*z + (1. - z)**2)


def Pgq_over(z): return TR


# the scale choice of alphaS
def scale_of_alphaS(t, z):
    return z * (1-z) * np.sqrt(t)

# Evaluate the frozen coupling / (2 pi), shared by the veto and its bound.
def alphaS_at_scale(scale, Qfreeze):
    if CMW == 'Factor':
        Nf = 5
        CMWFactor = np.exp(- (67 - 3 * np.pi**2 - 10/3 * Nf)/ (33 - 2*Nf) )
        scale *= CMWFactor
    elif CMW not in ('None', 'Linear'):
        raise ValueError("Unknown CMW scheme: " + str(CMW))
    scale = max(scale, Qfreeze)
    value = aS.alphasQ(scale)/2./np.pi
    if CMW == 'Linear':
        value *= 1 + Kg() * value
    return value

# Qfreeze is independent of the physical emission cutoff used by the shower.
def alphaS(t, z, Qfreeze, aSover):
    return alphaS_at_scale(scale_of_alphaS(t, z), Qfreeze)

# the analytical integral of t * Gamma over z
# EXTENDED: the default channel is the original q -> qg.
def tGamma(z, aSover, channel='qg'):
    if channel == 'qg':
        return -2.*aSover*CF*np.log1p(-z)
    if channel == 'gg':
        return aSover*CA*(np.log(z) - np.log1p(-z))
    if channel == 'qqbar':
        return aSover*TR*z
    raise ValueError('Unknown branching channel: ' + channel)


# the inverse of the function t*Gamma, given the overestimate for alphaS:
def inversetGamma(r, aSover, channel='qg'):
    if channel == 'qg':
        return 1. - np.exp(-0.5*r/CF/aSover)
    if channel == 'gg':
        return 1./(1. + np.exp(-r/CA/aSover))
    if channel == 'qqbar':
        return r/TR/aSover
    raise ValueError('Unknown branching channel: ' + channel)


# the overestimated upper and lower limit for the z integral:
def zp_over(t, cut, channel='qg'): return 1. - zm_over(t, cut, channel)


def zm_over(t, cut, channel='qg'):
    if channel == 'qg':
        return np.sqrt(cut**2/t)
    # Gluon channels require qtilde >= 4*pTmin. This is the stable form of
    # (1 - sqrt(1 - 4*pTmin/qtilde))/2.
    r = cut/np.sqrt(t)
    return 2.*r/(1. + np.sqrt(max(0., 1. - 4.*r)))


# set the overestimate of alphaS once and for all
def get_alphaS_over(Qfreeze):
    alphaS_over = alphaS_at_scale(0., Qfreeze)
    if debug: print('alpha_S overestimate set to', alphaS_over, 'for freeze scale=', Qfreeze, 'GeV')
    return alphaS_over

# get the momentum fraction candidate for the emission
def Get_zEmission(t, Qcut, R, aSover, channel='qg'):
    lower = tGamma(zm_over(t, Qcut, channel), aSover, channel)
    upper = tGamma(zp_over(t, Qcut, channel), aSover, channel)
    return inversetGamma(lower + R*(upper - lower), aSover, channel)

# calculate the transverse momentum of the emission
def Get_pTsq(t, z): return z**2 * (1-z)**2 * t

# calculate the virtual mass-squared of the emitting particle
def Get_mvirtsq(t,z): return z*(1-z) * t

# a function that calculates the emission scale given the initial scale Q, cutoff Qc and random number R
def Get_tEmission_direct(Q, Qcut, R, aSover, channel='qg'):
    fac_cutoff = 4. if channel == 'qg' else 16.
    if Q <= np.sqrt(fac_cutoff) * Qcut or R <= 0.:
        return Q**2, [], False
    upper = tGamma(zp_over(Q**2, Qcut, channel), aSover, channel)
    lower = tGamma(zm_over(Q**2, Qcut, channel), aSover, channel)
    if lower >= upper:
        if debug: print('\tEmission fails due upper < lower')
        return Q**2, [], False
    c = 1/(upper - lower)
    # get the actual evolution variable
    tEm_sol = Q**2 * R**c
    if math.isnan(tEm_sol) or tEm_sol < fac_cutoff*Qcut**2:
        if debug: print('\tEmission fails due to NaN tEm or tEm_sol < fac_cutoff*Qcut**2, tEm_sol=', tEm_sol)
        return Q**2, [], False
    return tEm_sol, [], True

# function that generates emissions:
def Generate_Emission(Q, Qcut, aSover, channel='qg'):
    fac_cutoff = 4. if channel == 'qg' else 16.
    kernel, kernel_over = {'qg': (Pqq, Pqq_over), 'gg': (Pgg, Pgg_over),
                           'qqbar': (Pgq, Pgq_over)}[channel]
    generated = True
    # generate random numbers
    R1 = random()
    R2 = random()
    R3 = random()
    R4 = random()
    # solve for the (candidate) emission scale:
    tEm, results, continueEvolution = Get_tEmission_direct(Q, Qcut, R1, aSover, channel)
    # if no solution is found then end branch
    if continueEvolution == False:
        zEm = 1.
        pTsqEm = 0.
        MsqEm = 0.
        if debug: print('continueEvolution is False')
        return tEm, zEm, pTsqEm, MsqEm, generated, continueEvolution
    if debug: print('\tcandidate emission scale, sqrt(tEm)=', np.sqrt(tEm))
    if tEm < fac_cutoff*Qcut**2:
        if debug: print('\t\temission REJECTED due to tEm < fac_cutoff*Qcut**2: tEm, Qcut=', tEm, Qcut)
        generated = False
    # calculate actual limits on z+, z- and check if they are consistent:
    zp_true = zp_over(tEm, Qcut, channel)
    zm_true = zm_over(tEm, Qcut, channel)
    if zm_true < 0 or zp_true < 0:
        if debug: print('\t\temission REJECTED due to zm_true < 0 or zp_true < 0: zm_true, zp_true=', zm_true, zp_true)
        generated = False
    if zm_true > zp_true:
        if debug: print('\t\temission REJECTED due to zm_true > zp_true: zm_true=', zm_true, 'zp_true=', zp_true)
        generated = False
    # get the (candidate) z of the emission
    zEm = Get_zEmission(Q**2, Qcut, R2, aSover, channel)
    if debug: print('\t\tcandidate momentum fraction, zEm=', zEm)
    # check that zEm is within allowed kinematic limits:
    if zEm < zm_true or zEm > zp_true:
        if debug: print('\t\temission REJECTED due to zEm < zm_true or zEm > zp_true: zEm=', zEm, 'zm_true', zm_true, 'zp=', zp_true)
        generated = False
    # get the transverse momentum
    pTsqEm = Get_pTsq(tEm, zEm)
    if debug: print('\t\tcandidate transverse momentum =', np.sqrt(pTsqEm))
    # check if below cutoff
    if pTsqEm < Qcut**2:
        if debug: print('\t\temission REJECTED due to pT < emission cutoff:', np.sqrt(pTsqEm), '<', Qcut)
        generated = False
    # now check the conditions to accept or reject the emission:
    # check if the transverse momentum is physical:
    if pTsqEm < 0.:
        if debug: print('\t\temission REJECTED due to negative pT**2=', pTsqEm)
        generated = False
    # compare the splitting function overestimate prob to a random number
    if kernel(zEm, tEm, Qc, aSover)/kernel_over(zEm) < R3:
        if debug: print('\t\temission REJECTED due to splitting function overestimate, p=', kernel(zEm, tEm, Qc, aSover)/kernel_over(zEm), 'R=', R3)
        generated = False
    else:
        if debug: print('\t\temission NOT rejected due to splitting function overestimate, p=', kernel(zEm, tEm, Qc, aSover)/kernel_over(zEm), 'R=', R3)
    # compare the alphaS overestimate prob to a random number
    coupling_probability = alphaS(tEm, zEm, Qc, aSover)/aSover
    if not 0. <= coupling_probability <= 1.:
        raise RuntimeError("Invalid alpha_s veto probability: " + str(coupling_probability)
                           + "; check the coupling freeze scale and overestimate")
    if coupling_probability < R4:
        if debug: print('\t\temission REJECTED due to alphaS overestimate, p=', coupling_probability, 'R=', R4)
        generated = False
    else:
        if debug: print('\t\temission NOT rejected due to alphaS overestimate, p=', coupling_probability, 'R=', R4)
    # get the virtual mass squared:
    MsqEm = Get_mvirtsq(tEm, zEm)
    if debug and generated == True:
        print('\t\t---> Emission accepted!')
    if generated == False: # rejected emission
        zEm = 1.
        pTsqEm = 0.
        MsqEm = 0.
        # NOTE: tEm continues from the rejected emission scale!
    # return all the variables for the emission
    return tEm, zEm, pTsqEm, MsqEm, generated, continueEvolution

# ADDED: finish the veto evolution of one channel before competing channels.
def Next_Emission(Q, Qcut, aSover, channel='qg', flavour=0):
    fac_cutoff = 4. if channel == 'qg' else 16.
    while Q > np.sqrt(fac_cutoff)*Qcut:
        tEm, zEm, pTsqEm, MsqEm, generated, continueEvolution = Generate_Emission(
            Q, Qcut, aSover, channel)
        if not continueEvolution:
            return None
        if generated:
            # The first five entries match the original Emissions array.
            # phi is sampled only after this channel wins the competition.
            return [tEm, zEm, np.sqrt(pTsqEm), MsqEm, 0., channel, flavour]
        Q = np.sqrt(tEm)  # Continue from a rejected trial, not the old scale.
    return None


# ADDED: the largest accepted scale wins, including one channel per flavour.
def Choose_Emission(pid, Q, Qcut, aSover):
    if pid != 21:
        return Next_Emission(Q, Qcut, aSover)
    if not gluon_splittings:
        return None
    Emission = Next_Emission(Q, Qcut, aSover, 'gg')
    for flavour in range(1, Nflavours + 1):
        candidate = Next_Emission(Q, Qcut, aSover, 'qqbar', flavour)
        if candidate is not None and (Emission is None or candidate[0] > Emission[0]):
            Emission = candidate
    return Emission


# ADDED: a branching history now needs to identify both daughter partons.
# Momenta are reconstructed later, as in pyresias_qtilde.py.
def make_parton(pid, color, anticolor, Q2start):
    return {'id': pid, 'color': color, 'anticolor': anticolor,
            'Q2start': Q2start, 'Emission': None, 'children': []}


def shower_partons(root):
    """Visit every parton without a particle-count or recursion-depth limit."""
    pending = [root]
    while pending:
        part = pending.pop()
        yield part
        pending.extend(reversed(part['children']))


def MakeDaughters(parent, Emission, colors):
    tEm, zEm, pT, MsqEm, phi, channel, flavour = Emission
    if channel == 'qg':
        ids = parent['id'], 21
    elif channel == 'gg':
        ids = 21, 21
    else:
        ids = flavour, -flavour
    left = make_parton(ids[0], 0, 0, zEm**2*tEm)
    right = make_parton(ids[1], 0, 0, (1.-zEm)**2*tEm)
    if channel == 'qqbar':
        left['color'], right['anticolor'] = parent['color'], parent['anticolor']
    else:
        new = next(colors)
        if channel == 'gg':
            hard, soft = (left, right) if zEm >= .5 else (right, left)
            if random() < .5:
                hard['color'], hard['anticolor'] = new, parent['anticolor']
                soft['color'], soft['anticolor'] = parent['color'], new
            else:
                hard['color'], hard['anticolor'] = parent['color'], new
                soft['color'], soft['anticolor'] = new, parent['anticolor']
        elif parent['id'] > 0:
            left['color'] = new
            right['color'], right['anticolor'] = parent['color'], new
        else:
            left['anticolor'] = new
            right['color'], right['anticolor'] = new, parent['anticolor']
    return [left, right]


# the function that performs the evolution of a single particle (e.g. from an LHE file)
def EvolveParticle(p, Qmin, Q2start, aSover, colors=None):
    # EXTENDED: replace the single quark line by a history with two daughters.
    # Each Emission still starts with [tEm, zEm, pT, MsqEm, phi], followed by
    # the channel and flavour. Only final leaves become outgoing particles.
    if colors is None:
        colors = count(max(p[7:9]) + 1)
    Emissions = make_parton(p[0], p[7], p[8], Q2start)
    pending = [Emissions]
    while pending:
        part = pending.pop()
        Emission = Choose_Emission(part['id'], np.sqrt(part['Q2start']), Qmin, aSover)
        if Emission is None:
            continue
        Emission[4] = (2*random() - 1)*np.pi
        part['Emission'] = Emission
        part['children'] = MakeDaughters(part, Emission, colors)
        # Evolve both daughters, including emitted gluons and secondary quarks.
        pending.extend(reversed(part['children']))
    if debug:
        PrintEmissions([part['Emission'] for part in shower_partons(Emissions)
                        if part['Emission'] is not None])
    return Emissions


# Shower an event (which consists of the "particles" array):
def Shower(particles, Qmin, aSover, statistics=None):
    # lists to store all emission and momenta information:
    AllMomenta = []
    JetMomenta = [] # to be used for global momentum conservation
    colors = count(max(int(c) for p in particles for c in p[7:9]) + 1)
    # The hard process is still massless e+e- -> q qbar.
    for p in particles:
        if abs(p[0]) == 11:
            AllMomenta.append(p)
        elif 0 < abs(p[0]) < 6 and p[1] == 1:
            # find the color partner and calculate the starting scale
            ppartner, Q2start = find_color_partner(p, particles)
            EmissionVariables = EvolveParticle(p, Qmin, Q2start, aSover, colors)
            # reconstruct in the Sudakov basis, then rotate to the lab frame
            Momenta = reconstructSudakov(p, ppartner, EmissionVariables)
            RotatedMomenta = RotateMomentaLab(p, Momenta)
            AllMomenta.extend(RotatedMomenta)
            JetMomenta.append([p, RotatedMomenta])
            if statistics is not None:
                for part in shower_partons(EmissionVariables):
                    if part['Emission'] is not None:
                        channel = part['Emission'][5]
                        statistics[channel] = statistics.get(channel, 0) + 1
    return AllMomenta, JetMomenta


# find the color partner of a given particle part in particles
def find_color_partner(part, particles):
    """Returns the color partner of particle part"""
    partner = []
    for pc in particles:
        if part[7] == pc[8] and part[8] == pc[7]:
            partner = pc
            if debug: print('Color partner of particle', part, 'found:', pc)
    if len(partner) != 0:
        # set the starting scale as just the invariant mass of the sum of the partners
        Q2start = (part[5]+partner[5])**2 - ((part[2]+partner[2])**2 + (part[3]+partner[3])**2 + (part[4]+partner[4])**2)
    else:
        Q2start = 0
    return partner, Q2start


# reconstruct the particles in the Sudakov basis (back-to-back frame)
def reconstructSudakov(pin, nin, EmissionVariables):
    Momenta = []
    # go to a frame where the progenitor is moving in the z direction:
    pmag = np.sqrt(pin[2]**2 + pin[3]**2 + pin[4]**2)
    p = [pin[0], pin[1], 0, 0, pmag, pmag, 0]
    nmag = np.sqrt(nin[2]**2 + nin[3]**2 + nin[4]**2)
    n = [nin[0], nin[1], 0, 0, -nmag, nmag, 0]
    pdotn = dot4vec(p, n)
    # EXTENDED: propagate the same alpha and qT equations to both daughters.
    # Each pending entry holds (parton, alpha, transverse four-vector).
    pending = [(EmissionVariables, 1., [0, 0, 0., 0., 0, 0])]
    while pending:
        part, alpha_prime, qT_prime = pending.pop()
        Emission = part['Emission']
        if Emission is not None:
            z = Emission[1]
            pT = Emission[2]
            phi = Emission[4]
            alpha = alpha_prime*(1-z)
            alpha_prime *= z
            kT = [pT*np.cos(phi), pT*np.sin(phi), 0, 0]
            qT = [0, 0, qT_prime[2]*(1-z) - kT[0], qT_prime[3]*(1-z) - kT[1], 0, 0]
            qT_prime = [0, 0, qT_prime[2]*z + kT[0], qT_prime[3]*z + kT[1], 0, 0]
            left, right = part['children']
            pending.append((left, alpha_prime, qT_prime))
            pending.append((right, alpha, qT))
            continue
        # A final leaf is on shell; intermediate virtualities are sums of daughters.
        if alpha_prime <= 0. or not np.isfinite(alpha_prime):
            raise KinematicsError('Invalid Sudakov light-cone momentum')
        pTisq = qT_prime[2]**2 + qT_prime[3]**2
        beta = pTisq/(2*alpha_prime*pdotn)
        px = alpha_prime*p[2] + beta*n[2] + qT_prime[2]
        py = alpha_prime*p[3] + beta*n[3] + qT_prime[3]
        pz = alpha_prime*p[4] + beta*n[4]
        E = alpha_prime*p[5] + beta*n[5]
        Momenta.append([part['id'], 1, px, py, pz, E, 0,
                        part['color'], part['anticolor']])
    return Momenta


def main(argv=None):
    return run_shower(sys.modules[__name__], argv, qtilde=True, full=True)


if __name__ == "__main__":
    raise SystemExit(main())
