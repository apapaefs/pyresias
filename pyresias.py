"""Introductory massless quark shower with global recoil."""
from random import random
import math
import sys
import numpy as np
from prettytable import PrettyTable
from alphaS import alphaS, CF
from shower_cli import run_shower
from kinematics import RotateMomentaLab

debug = False
Qc = .935

# function to print the emission information once they have been genrated:
def PrintEmissions(EmissionsArray):
    tbl = PrettyTable(["#", "Evo scale [GeV]", '1-z', 'pT [GeV]', 'virt. mass in a->bc [GeV]'])
    for i in range(len(EmissionsArray)):
        tbl.add_row([i, EmissionsArray[i][0], 1-EmissionsArray[i][1], EmissionsArray[i][2], EmissionsArray[i][3]])
    print(tbl)

# function to print the momenta information once they have been genrated:
def PrintMomenta(MomentaArray):
    tbl = PrettyTable(["n", "id", "status", 'px [GeV]', 'py [GeV]', 'pz [GeV]', 'E [GeV]', 'm [GeV]'])
    for i in range(len(MomentaArray)):
        tbl.add_row([i, MomentaArray[i][0], MomentaArray[i][1], MomentaArray[i][2], MomentaArray[i][3], MomentaArray[i][4], MomentaArray[i][5], MomentaArray[i][6]])
    print(tbl)


# initialize alphaS class: pass the value of alphaS at mz, and mz
aS = alphaS(0.118, 91.1876)

#################################################

# the q -> q + g splitting function
def Pqq(z): return CF * (1. + z**2)/(1.-z)
    
# the q -> q + g splitting function *overestimate* 
def Pqq_over(z): return 2.*CF/(1.-z)

# the scale choice of alphaS 
def scale_of_alphaS(t, z):
    return z * (1-z) * np.sqrt(t)

# Return the local running coupling divided by 2 pi, frozen at Qcut.
def alphaS(t, z, Qcut, aSover):
    scale = scale_of_alphaS(t, z)
    if scale < Qcut:
        return aS.alphasQ(Qcut)/2./np.pi
    return aS.alphasQ(scale)/2./np.pi

# the analytical integral of t * Gamma over z 
def tGamma(z, aSover):
    return -2.*aSover*CF*np.log1p(-z) 

# the inverse of the function t*Gamma, given the overestimate for alphaS:
def inversetGamma(r, aSover):
    return 1. - np.exp(- 0.5*r/CF/aSover)

# the overestimated upper and lower limit for the z integral:
def zp_over(t, Qcut): return 1.-np.sqrt(Qcut**2/t)
def zm_over(t, Qcut): return np.sqrt(Qcut**2/t)

# set the overestimate of alphaS once and for all
def get_alphaS_over(Qcut):
    alphaS_over = aS.alphasQ(Qcut)/2./np.pi
    if debug: print('alpha_S overestimate set to', alphaS_over, 'for scale=', Qcut, 'GeV')
    return alphaS_over

# get the momentum fraction candidate for the emission
def Get_zEmission(t, Qcut, R, aSover): return inversetGamma( tGamma(zm_over(t, Qcut), aSover) + R * ( tGamma(zp_over(t, Qcut), aSover) - tGamma(zm_over(t, Qcut), aSover)), aSover)
    
# calculate the transverse momentum of the emission
def Get_pTsq(t, z): return z**2 * (1-z)**2 * t

# calculate the virtual mass-squared of the emitting particle
def Get_mvirtsq(t,z): return z*(1-z) * t

# a function that calculates the emission scale given the initial scale Q, cutoff Qc and random number R
def Get_tEmission_direct(Q, Qcut, R, aSover):
    if Q <= 2. * Qcut or R <= 0.:
        return Q**2, [], False
    upper = tGamma(zp_over(Q**2, Qcut), aSover)
    lower = tGamma(zm_over(Q**2, Qcut), aSover)
    if lower >= upper:
        if debug: print('\tEmission fails due upper < lower')
        return Q**2, [], False
    c = 1/(upper - lower)
    # get the actual evolution variable
    tEm_sol = Q**2 * R**c
    if math.isnan(tEm_sol) or tEm_sol < 4*Qcut**2:
        if debug: print('\tEmission fails due to NaN tEm or tEm_sol < 4*Qcut**2, tEm_sol=', tEm_sol)
        return Q**2, [], False
    return tEm_sol, [], True

# function that generates emissions:
def Generate_Emission(Q, Qcut, aSover):
    generated = True
    # generate random numbers
    R1 = random()
    R2 = random()
    R3 = random()
    R4 = random()
    # solve for the (candidate) emission scale:
    tEm, results, continueEvolution = Get_tEmission_direct(Q, Qcut, R1, aSover)
    # if no solution is found then end branch
    if continueEvolution == False:
        zEm = 1.
        pTsqEm = 0.
        MsqEm = 0.
        if debug: print('continueEvolution is False')
        return tEm, zEm, pTsqEm, MsqEm, generated, continueEvolution
    if debug: print('\tcandidate emission scale, sqrt(tEm)=', np.sqrt(tEm))
    if tEm < 4*Qcut**2:
        if debug: print('\t\temission REJECTED due to tEm < 4*Qcut**2: tEm, Qcut=', tEm, Qcut)
        generated = False
    # calculate actual limits on z+, z- and check if they are consistent:
    zp_true = zp_over(tEm, Qcut)
    zm_true = zm_over(tEm, Qcut)
    if zm_true < 0 or zp_true < 0:
        if debug: print('\t\temission REJECTED due to zm_true < 0 or zp_true < 0: zm_true, zp_true=', zm_true, zp_true)
        generated = False
    if zm_true > zp_true:
        if debug: print('\t\temission REJECTED due to zm_true > zp_true: zm_true=', zm_true, 'zp_true=', zp_true)
        generated = False
    # get the (candidate) z of the emission
    zEm = Get_zEmission(Q**2, Qcut, R2, aSover)
    if debug: print('\t\tcandidate momentum fraction, zEm=', zEm)
    # check that zEm is within allowed limits:
    if zEm < zm_true or zEm > zp_true:
        if debug: print('\t\temission REJECTED due to zEm < zm_true or zEm > zp_true: zEm=', zEm, 'zm_true', zm_true, 'zp=', zp_true)
        generated = False
    # get the transverse momentum 
    pTsqEm = Get_pTsq(tEm, zEm)
    if debug: print('\t\tcandidate transverse momentum =', np.sqrt(pTsqEm))
    # check if below cutoff
    if pTsqEm < Qcut**2:
        if debug: print('\t\temission REJECTED due to pT <  pTmin:', np.sqrt(pTsqEm), '<', Qcut)
        generated = False
    # now check the conditions to accept or reject the emission:
    # check if the transverse momentum is physical:
    if pTsqEm < 0.:
        if debug: print('\t\temission REJECTED due to negative pT**2=', pTsqEm)
        generated = False
    # compare the splitting function overestimate prob to a random number
    if Pqq(zEm)/Pqq_over(zEm) < R3:
        if debug: print('\t\temission REJECTED due to splitting function overestimate, p=', Pqq(zEm)/Pqq_over(zEm), 'R=', R3)
        generated = False
    else:
        if debug: print('\t\temission NOT rejected due to splitting function overestimate, p=', Pqq(zEm)/Pqq_over(zEm), 'R=', R3)
    # compare the alphaS overestimate prob to a random number
    coupling_probability = alphaS(tEm, zEm, Qcut, aSover)/aSover
    if not 0. <= coupling_probability <= 1.:
        raise RuntimeError("Invalid alpha_s veto probability: " + str(coupling_probability))
    if coupling_probability < R4:
        if debug: print('\t\temission REJECTED due to alphaS overestimate: alphaS, aSover, p=', 2*np.pi*alphaS(tEm, zEm, Qcut, aSover), 2*np.pi*aSover, alphaS(tEm, zEm, Qcut, aSover)/aSover, 'R=', R4)
        generated = False
    else:
        if debug: print('\t\temission NOT rejected due to alphaS overestimate: alphaS, aSover, p=', 2*np.pi*alphaS(tEm, zEm, Qcut, aSover), 2*np.pi*aSover, alphaS(tEm, zEm, Qcut, aSover)/aSover, 'R=', R4)
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

# the function that performs the evolution of a single particle (e.g. from an LHE file)
# returns a list of all outgoing particles
def EvolveParticle(p, Qmin, aSover):
    # the minimum evolution scale
    tEm_min = Qmin**2
    # counter for the number of emissions:
    Nem = 0
    # array to store emission info:
    Emissions = []
    # array to store momenta of outgoing particles:
    Momenta = []
    fac_cutoff = 4. # actual cutoff = fac_cutoff * Qc**2
    # star the evolution
    tEm = p[5]**2 # initial value of the evolution variable = the energy of the particle
    zEm = 1 # initial value of the momentum fraction
    # initial magnitude of the quark momentum
    pmag = np.sqrt(p[2]**2 + p[3]**2 + p[4]**2)
    if debug: print('generating evolution for E=', p[5], 'GeV\n')
    # continue the evolution while we are above the cutoff:
    while np.sqrt(tEm)*zEm > np.sqrt(fac_cutoff*tEm_min):
        # evolve:
        tEm, zEm, pTsqEm, MsqEm, generatedEmission, continueEvolution = Generate_Emission(np.sqrt(tEm)*zEm, np.sqrt(tEm_min), aSover)
        # if the solver could not find a solution, end the evolution
        if continueEvolution == False:
            if debug:
                print('no further emissions, evolution ended')
                print('total number of emissions=', Nem)
                print('\n')
                print('-----')
                print('Emissions table:')
                PrintEmissions(Emissions)
            Momenta.append([p[0], 1, 0, 0, pmag, pmag, 0])
            return Emissions, Momenta
        # if we have already passed the cutoff this emission does not count
        # this will also terminate the evolution
        if tEm < fac_cutoff*tEm_min: 
            if debug: print('\t\tXX->emission rejected at sqrt(t)=', np.sqrt(tEm), 'since it is below cutoff')
            zEm = 1.
            pTsqEm = 0.
            if debug: print('total number of emissions=', Nem)
            Momenta.append([p[0], 1, 0, 0, pmag, pmag, 0])
            return Emissions, Momenta
        # if the emission was successful, append to the Emissions and Momenta lists and continue
        if zEm != 1.0:
            pT = np.sqrt(pTsqEm)
            Emissions.append([np.sqrt(tEm), zEm, pT, np.sqrt(MsqEm)])
            # generate the momenta of the outgoing gluons with respect to the quark direction
            # random phi angle
            phi = (2*random() - 1)*np.pi
            Ei = np.sqrt( (1-zEm)**2 * pmag**2 + pT**2 )
            Momenta.append([21, 1, pT*np.cos(phi), pT*np.sin(phi), (1-zEm)*pmag, Ei, 0])
            # rescale the magnitude of the parent particle by z
            pmag = zEm * pmag
            if debug: print('\t->successful emission at sqrt(t)=', np.sqrt(tEm), 'z=', zEm, 'pT=', np.sqrt(pTsqEm), 'mVirt=', np.sqrt(MsqEm))
            Nem = Nem + 1
    if debug:
        print('no further emissions, evolution ended')
        print('total number of emissions=', Nem)
        print('\n')
        print('-----')
        print('Emissions table:')
        PrintEmissions(Emissions)
    # add the magnitude of the quark with respect to its original direction:
    Momenta.append([p[0], 1, 0, 0, pmag, pmag, 0])
    return Emissions, Momenta


# Shower an event (which consists of the "particles" array):
def Shower(particles, Qmin, aSover):
    # lists to store all emission and momenta information:
    AllMomenta = []
    JetMomenta = [] # to be used for global momentum conservation
    # Find the colored particles and shower them down to Qmin
    # For now this only works on *FINAL STATE QUARKS AND BEAM PARTICLES CAN ONLY BE ELECTRONS/POSITRONS*!
    for p in particles:
        if abs(p[0]) == 11:
           AllMomenta.append(p)
        elif abs(p[0])>0 and abs(p[0])<6 and p[1]==1: # treat quarks up to the b-quark as massless, ignore top quarks for now
            if debug:
                print('Showering quark:', p[0])
            Emissions, Momenta = EvolveParticle(p, Qmin, aSover)
            # rotate the momenta to align with the direction of the particle in lab frame
            RotatedMomenta = RotateMomentaLab(p, Momenta)
            # print
            # PrintMomenta(RotatedMomenta)
            # append to array:
            for Mom in RotatedMomenta:
                AllMomenta.append(Mom)
            # append momenta of the "jet" before and after into the list
            JetMomenta.append([p, RotatedMomenta])
    return AllMomenta, JetMomenta



def main(argv=None):
    return run_shower(sys.modules[__name__], argv, qtilde=False)


if __name__ == "__main__":
    raise SystemExit(main())
