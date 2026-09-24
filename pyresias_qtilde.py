"""Massless angular-ordered shower in the Sudakov basis."""
from random import random
import math
import sys
import numpy as np
from prettytable import PrettyTable
from alphaS_HW import alphaS, CF
from shower_cli import run_shower
from kinematics import RotateMomentaLab, dot4vec

debug = False
Qc = .935
pTmin = .900

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
def tGamma(z, aSover):
    return -2.*aSover*CF*np.log1p(-z) 

# the inverse of the function t*Gamma, given the overestimate for alphaS:
def inversetGamma(r, aSover):
    return 1. - np.exp(- 0.5*r/CF/aSover)

# the overestimated upper and lower limit for the z integral:
def zp_over(t, cut): return 1.-np.sqrt(cut**2/t)
def zm_over(t, cut): return np.sqrt(cut**2/t)

# set the overestimate of alphaS once and for all
def get_alphaS_over(Qfreeze):
    alphaS_over = alphaS_at_scale(0., Qfreeze)
    if debug: print('alpha_S overestimate set to', alphaS_over, 'for freeze scale=', Qfreeze, 'GeV')
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
    if Pqq(zEm, tEm, Qc, aSover)/Pqq_over(zEm) < R3:
        if debug: print('\t\temission REJECTED due to splitting function overestimate, p=', Pqq(zEm, tEm, Qc, aSover)/Pqq_over(zEm), 'R=', R3)
        generated = False
    else:
        if debug: print('\t\temission NOT rejected due to splitting function overestimate, p=', Pqq(zEm, tEm, Qc, aSover)/Pqq_over(zEm), 'R=', R3)
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

# the function that performs the evolution of a single particle (e.g. from an LHE file)
# returns a list of all outgoing particles
def EvolveParticle(p, Qmin, Q2start, aSover):
    # the minimum evolution scale
    tEm_min = Qmin**2
    # counter for the number of emissions:
    Nem = 0
    # array to store emission info:
    Emissions = []
    fac_cutoff = 4. # actual cutoff = fac_cutoff * Qc**2
    # star the evolution
    tEm = Q2start # initial value of the evolution variable = COM energy in this case
    zEm = 1.0 # initial value of the momentum fraction
    if debug: print('generating evolution for starting scale sqrt(t)=', np.sqrt(Q2start), 'GeV\n')
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
            return Emissions
        # if we have already passed the cutoff this emission does not count
        # this will also terminate the evolution
        if tEm < fac_cutoff*tEm_min: 
            if debug: print('\t\tXX->emission rejected at sqrt(t)=', np.sqrt(tEm), 'since it is below cutoff')
            zEm = 1.
            pTsqEm = 0.
            if debug: print('total number of emissions=', Nem)
            return Emissions
        # if the emission was successful, append to the Emissions and Momenta lists and continue
        if zEm != 1.0:
            pT = np.sqrt(pTsqEm)
            # random phi angle
            phi = (2*random() - 1)*np.pi
            Emissions.append([tEm, zEm, pT, MsqEm, phi])
            Nem = Nem + 1
    if debug:
        print('no further emissions, evolution ended')
        print('total number of emissions=', Nem)
        print('-----')
        print('Emissions table:')
        PrintEmissions(Emissions)
    return Emissions


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
            # find the color partner and calculate the starting scale
            ppartner, Q2start = find_color_partner(p, particles)
            EmissionVariables = EvolveParticle(p, Qmin, Q2start, aSover)
            # since we are already in the back-to-back frame in e+e- -> qqbar, and we are considering light quarks,
            # ppartner is already the n vector (reference vector) and p is p in the Sudakov basis.
            # reconstruct momenta according to the Sudakov decomposition from the emission variables, p and n
            Momenta = reconstructSudakov(p,ppartner,EmissionVariables)
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
    Momenta = [] # array to hold the momenta
    alphas = [] # the alphas of the emitted gluons
    betas = [] # the betas of the emitted gluons
    betas_prime = [] # the betas of the quark
    alphas_prime = [] # the alphas of the evolving quark
    qTs_prime = [] # the transverse momentum 4-vector of the quark
    qTs = [] # the transverse momentum 4-vector of the gluons
    # initial values:
    alpha_prime = 1 # the alpha of the evolving quark
    qT_prime = [0,0,0,0,0,0] # initial qT of quark is zero
    # go to a frame where the progenitor is moving in the z direction:
    pmag = np.sqrt(pin[2]**2 + pin[3]**2 + pin[4]**2)
    p = [pin[0], pin[1], 0, 0, pmag, pmag, 0]
    nmag = np.sqrt(nin[2]**2 + nin[3]**2 + nin[4]**2)
    n = [nin[0], nin[1], 0, 0, -nmag, nmag, 0]
    pdotn = dot4vec(p,n) # get p.n
    if len(EmissionVariables) == 0:
        return [p]
    
    # get the information generated in each emission
    for Emission in EmissionVariables:
        z = Emission[1] # get the momentum fraction of the emission
        pT = Emission[2] # get the pT of the emission
        phi = Emission[4] # get the phi of the emission
        
        # get the alphas for the quark and emitted gluon: 
        alpha = alpha_prime * (1-z) # alpha of emitted gluon
        alpha_prime *= z # alpha of evolving quark
        
        # append to lists:
        alphas.append(alpha) 
        alphas_prime.append(alpha_prime)
 
        # calculate the qT 4-vectors
        kT = [pT*np.cos(phi), pT*np.sin(phi), 0, 0] # (px, py, pz, E)
        qT = [0, 0, qT_prime[2]*(1-z) - kT[0], qT_prime[3]*(1-z) - kT[1], 0, 0] # (0, 0, px, py, pz, E)
        qT_prime =  [0, 0, qT_prime[2]*z + kT[0], qT_prime[3]*z + kT[1], 0, 0] # (0, 0, px, py, pz, E)
        qTs.append(qT)
        qTs_prime.append(qT_prime)
        
    # at this point, alphas and qTs have been calculated for each particle 
    # find the betas:
    for i in range(len(alphas)):
        pTisq = qTs[i][2]**2+qTs[i][3]**2
        beta = pTisq / (2 * alphas[i] * pdotn)
        pTisq_prime = qTs_prime[i][2]**2 + qTs_prime[i][3]**2 
        beta_prime = pTisq_prime / (2 * alphas_prime[i] * pdotn)
        # append
        betas.append(beta)
        betas_prime.append(beta_prime)
    # find the components of the 4-momenta of the emitted gluons
    for i in range(len(alphas)):
        px = alphas[i] * p[2] + betas[i] * n[2] + qTs[i][2]
        py = alphas[i] * p[3] + betas[i] * n[3] + qTs[i][3]
        pz = alphas[i] * p[4] + betas[i] * n[4]
        E = alphas[i] * p[5] + betas[i] * n[5]
        #print('Invariant mass SQUARED of emitted gluon=', E**2 - px**2 - py**2 - pz**2)
        # append to list:
        Momenta.append([21, 1, px, py, pz, E, 0])
    # find the final momentum of the quark: (the last instance)
    px = alphas_prime[-1] * p[2] + betas_prime[-1] * n[2] + qTs_prime[-1][2]
    py = alphas_prime[-1] * p[3] + betas_prime[-1] * n[3] + qTs_prime[-1][3]
    pz = alphas_prime[-1] * p[4] + betas_prime[-1] * n[4]
    E = alphas_prime[-1] * p[5] + betas_prime[-1] * n[5]
    Momenta.append([pin[0], 1, px, py, pz, E, 0])
    
    return Momenta
    
         



def main(argv=None):
    return run_shower(sys.modules[__name__], argv, qtilde=True)


if __name__ == "__main__":
    raise SystemExit(main())
