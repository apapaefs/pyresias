from random import * # random numbers
import os, subprocess # to check and create directories 
import math # python math
import numpy as np # numerical python
import scipy # scientific python
from scipy import optimize # for numerical solution of equations
from matplotlib import pyplot as plt # plotting
import matplotlib.gridspec as gridspec # more plotting 
from prettytable import PrettyTable # pretty printing of tables
from tqdm import tqdm # display progress
from optparse import OptionParser # command line parameters
from scipy.integrate import quad # for numerical integrals
from LHEReader import * # read LHE files
from HEPMCWriter import * # write HepMC files using pyhepmc
from LHEWriter import * # write LHE file
from alphaS_HW import * # alphaS at LO and NLO

################################################
print('\nPyresias-qTilde: a toy parton shower\n')
# a simple toy q -> q+g parton shower
#################################################

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


#############
# SWITCHES: #
#############

# switch to print information or not:
debug = False

# print the showered events:
printevents = False

# the input file name:
inputfile = ''

# output file name for the hepmc file
outputfile = ''

# RUN OPTIONS (defaults):

# how many events to shower
Nshower = 1E99

# the cutoff scale in GeV, 0.935 GeV matches Herwig 7 
Qc = 0.935
# minimum pT in the shower, matches Herwig 7
pTmin = 0.900

##########################
# COMMAND LINE ARGUMENTS #
##########################

parser = OptionParser(usage="%prog [options] [inputfile]", version='Pyresias 0.2')

parser.add_option("-d", "--debug", dest="debug", default=False, action="store_true",
                  help="Print debugging to screen")

parser.add_option("-p", "--printevents", dest="printevents", default=False, action="store_true",
                  help="Print showered events to screen")

parser.add_option("-n", "--nshower", dest="nshower", default=Nshower,
                  help="Set the number of events to shower")

parser.add_option("-c", dest="Qc", default=Qc,
                  help="Set the cutoff scale for the evolution")

parser.add_option("-o", dest="output", default=outputfile,
                  help="Set the output file name")


# parse the command line arguments
(options, args) = parser.parse_args()

# set command line arguments 
debug = options.debug
printevents = options.printevents
Nshower = int(options.nshower)
Qc = float(options.Qc)
outputfile = str(options.output)

if len(sys.argv) < 2: parser.error("An input file is required!")

inputfile = sys.argv[1]

if outputfile == '': outputfile = inputfile.replace('.lhe','').replace('.gz','') + '.hepmc'

#################################################

# initialize alphaS class: pass the value of alphaS at mz, and mz
aS = alphaS(0.1074, 91.1876) # alphaS(MZ) = 0.126 corresponds to 0.118 in MSbar scheme

# CMW scheme:
CMW = 'None' # 'Linear' or 'Factor' or 'None' 

#################################################

# the Kg function for the CMW Scheme
def Kg():
    Nf=5 # 5-flavors
    return 3.*(67./18.-1./6.*np.pi**2)-5./9.*Nf

# the q -> q + g splitting function
def Pqq(z, t, Qcut, aSover):
    if CMW == 'Linear' or CMW == 'Factor':
        aS = alphaS(t, z, Qcut, aSover)
        return CF * (1 + aS/2/np.pi * Kg() + z**2) / (1.-z)
    elif CMW == 'None':
        return CF * (1. + z**2)/(1.-z)
    
# the q -> q + g splitting function *overestimate* 
def Pqq_over(z): return 2.*CF/(1.-z)

# the scale choice of alphaS 
def scale_of_alphaS(t, z):
    return z * (1-z) * math.sqrt(t)

# return the true alphaS using the PDF alphaS over 2 pi
def alphaS(t, z, Qcut, aSover):
    scale = scale_of_alphaS(t, z)
    # if scale is < Qcut, reset scale to Qcut
    if scale < Qcut and CMW == 'None' or CMW == 'Linear':
        scale = Qcut
    if CMW == 'Linear':
        CMWFactor = 1 + Kg() * aS.alphasQ(scale)/2./math.pi
        return aS.alphasQ(scale)/2./math.pi * CMWFactor
    elif CMW == 'Factor':
        Nf = 5
        CMWFactor = np.exp(- (67 - 3 * np.pi**2 - 10/3 * Nf)/ (33 - 2*Nf) )
        scale *= CMWFactor
        if scale < Qcut:
            scale = Qcut
        return aS.alphasQ(scale)/2./math.pi
    elif CMW == 'None':
        return aS.alphasQ(scale)/2./math.pi

    

# the analytical integral of t * Gamma over z 
def tGamma(z, aSover):
    return -2.*aSover*CF*np.log1p(-z) 

# the inverse of the function t*Gamma, given the overestimate for alphaS:
def inversetGamma(r, aSover):
    return 1. - math.exp(- 0.5*r/CF/aSover)

# the overestimated upper and lower limit for the z integral:
def zp_over(t, cut): return 1.-math.sqrt(cut**2/t)
def zm_over(t, cut): return math.sqrt(cut**2/t)

# set the overestimate of alphaS once and for all
def get_alphaS_over(Qcut):
    minscale = Qcut # the minimum scale^2 available to the PDF
    if minscale < Qcut:
        scale = minscale
    else:
        scale = Qcut
    if CMW == 'Linear':
        CMWFactor = 1 + Kg() * aS.alphasQ(scale)/2./math.pi
        alphaS_over = aS.alphasQ(scale)/2./math.pi * CMWFactor
    elif CMW == 'Factor':
        Nf = 5
        CMWFactor = np.exp(- (67 - 3 * np.pi**2 - 10/3 * Nf)/ (33 - 2*Nf) )
        scale *= CMWFactor
        alphaS_over = aS.alphasQ(scale)/2./math.pi
    elif CMW == 'None':
        alphaS_over = aS.alphasQ(scale)/2./math.pi
    if debug: print('alpha_S overestimate set to', alphaS_over, 'for scale=', scale, 'GeV')
    return alphaS_over

# get the momentum fraction candidate for the emission
def Get_zEmission(t, Qcut, R, aSover): return inversetGamma( tGamma(zm_over(t, Qcut), aSover) + R * ( tGamma(zp_over(t, Qcut), aSover) - tGamma(zm_over(t, Qcut), aSover)), aSover)
    
# calculate the transverse momentum of the emission
def Get_pTsq(t, z): return z**2 * (1-z)**2 * t

# calculate the virtual mass-squared of the emitting particle
def Get_mvirtsq(t,z): return z*(1-z) * t

# a function that calculates the emission scale given the initial scale Q, cutoff Qc and random number R
def Get_tEmission_direct(Q, Qcut, R, aSover):
    upper = tGamma(zp_over(Q**2, Qcut), aSover)
    lower = tGamma(zm_over(Q**2, Qcut), aSover)
    if lower > upper:
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
    if debug: print('\tcandidate emission scale, sqrt(tEm)=', math.sqrt(tEm))
    if tEm < 4*Qcut**2:
        if debug: print('\t\temission REJECTED due to tEm < 4*Qcut**2: tEm, Qcut=', tEm, Qcut)
        generated = False
    # calculate actual limits on z+, z- and check if they are consistent:
    zp = zp_over(tEm, Qcut)
    zm = zm_over(tEm, Qcut)
    if zm < 0 or zp < 0:
        if debug: print('\t\temission REJECTED due to zm < 0 or zp < 0: zm,zp=', zm, zp)
        generated = False
    if zm > zp:
        if debug: print('\t\temission REJECTED due to zm > zp: zm=', zm, 'zp=', zp)
        generated = False
    # get the (candidate) z of the emission
    zEm = Get_zEmission(tEm, Qcut, R2, aSover)
    if debug: print('\t\tcandidate momentum fraction, zEm=', zEm)
    # check that zEm is within allowed limits:
    if zEm < zm or zEm > zp:
        if debug: print('\t\temission REJECTED due to zEm < zm or zEm > zp: zEm=', zEm, 'zm=', zm, 'zp=', zp)
        generated = False
    if zEm < zm or zEm > zp:
        if debug: print('\t\temission REJECTED due to zEm < zm or zEm > zp: zEm=', zEm, 'zm=', zm, 'zp=', zp)
        generated = False
    # get the transverse momentum 
    pTsqEm = Get_pTsq(tEm, zEm)
    if debug: print('\t\tcandidate transverse momentum =', np.sqrt(pTsqEm))
    # check if below cutoff
    if pTsqEm < pTmin**2:
        if debug: print('\t\temission REJECTED due to pT <  pTmin:', np.sqrt(pTsqEm), '<', pTmin)
        generated = False
    # now check the conditions to accept or reject the emission:
    # check if the transverse momentum is physical:
    if pTsqEm < 0.:
        if debug: print('\t\temission REJECTED due to negative pT**2=', pTsqEm)
        generated = False
    # compare the splitting function overestimate prob to a random number
    if Pqq(zEm, tEm, Qcut, aSover)/Pqq_over(zEm) < R3:
        if debug: print('\t\temission REJECTED due to splitting function overestimate, p=', Pqq(zEm, tEm, Qcut, aSover)/Pqq_over(zEm), 'R=', R3)
        generated = False
    else:
        if debug: print('\t\temission NOT rejected due to splitting function overestimate, p=', Pqq(zEm, tEm, Qcut, aSover)/Pqq_over(zEm), 'R=', R3)
    # compare the alphaS overestimate prob to a random number
    if alphaS(tEm, zEm, Qcut, aSover)/aSover < R4:
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
    # array to store momenta of outgoing particles:
    Momenta = []
    fac_cutoff = 4. # actual cutoff = fac_cutoff * Qc**2
    # star the evolution
    tEm = Q2start # initial value of the evolution variable = COM energy in this case
    zEm = 1 # initial value of the momentum fraction
    if debug: print('generating evolution for starting scale sqrt(t)=', np.sqrt(Q2start), 'GeV\n')
    # continue the evolution while we are above the cutoff:
    while np.sqrt(tEm)*zEm > np.sqrt(fac_cutoff*tEm_min):
        # evolve:
        tEm, zEm, pTsqEm, MsqEm, generatedEmission, continueEvolution = Generate_Emission(np.sqrt(tEm)*zEm, math.sqrt(tEm_min), aSover)
        # if the solver could not find a solution, end the evolution
        if continueEvolution == False:
            if debug:
                print('no further emissions, evolution ended')
                print('total number of emissions=', Nem)
                print('\n')
                print('-----')
                print('Emissions table:')
                PrintEmissions(Emissions)
            #Momenta.append([p[0], 1, 0, 0, pmag, pmag, 0])
            return Emissions
        # if we have already passed the cutoff this emission does not count
        # this will also terminate the evolution
        if tEm < fac_cutoff*tEm_min: 
            if debug: print('\t\tXX->emission rejected at sqrt(t)=', math.sqrt(tEm), 'since it is below cutoff')
            zEm = 1.
            pTsqEm = 0.
            QsqEm = 0.
            if debug: print('total number of emissions=', Nem)
            return Emissions
        # if the emission was successful, append to the Emissions and Momenta lists and continue
        if zEm != 1.0:
            pT = math.sqrt(pTsqEm)
            # random phi angle
            phi = (2*random() - 1)*np.pi
            Emissions.append([tEm, zEm, pT, MsqEm, phi])
            # generate the momenta of the outgoing gluons with respect to the quark direction
            #Ei = np.sqrt( (1-zEm)**2 * pmag**2 + pT**2 )
            #Momenta.append([21, 1, pT*np.cos(phi), pT*np.sin(phi), (1-zEm)*pmag, Ei, 0])
            # rescale the magnitude of the parent particle by z
            #pmag = zEm * pmag
            #if debug: print('\t->successful emission at sqrt(t)=', math.sqrt(tEm), 'z=', zEm, 'pT=', math.sqrt(pTsqEm), 'mVirt=', math.sqrt(MsqEm))
            Nem = Nem + 1
    if debug:
        print('no further emissions, evolution ended')
        print('total number of emissions=', Nem)
        print('-----')
        print('Emissions table:')
        PrintEmissions(Emissions)
    # add the magnitude of the quark with respect to its original direction:
    #Momenta.append([p[0], 1, 0, 0, pmag, pmag, 0])
    return Emissions#, Momenta


# Shower an event (which consists of the "particles" array):
def Shower(particles, Qmin, aSover):
    # lists to store all emission and momenta information:
    AllEmissions = []
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
    qsqs = [] # the virtualities 
    # initial values:
    alpha_prime = 1 # the alpha of the evolving quark
    qT_prime = [0,0,0,0,0,0] # initial qT of quark is zero
    qT = [0,0,0,0,0,0] # initial qT is zero
    # go to a frame where the progenitor is moving in the z direction:
    pmag = np.sqrt(pin[2]**2 + pin[3]**2 + pin[4]**2)
    p = [pin[0], pin[1], 0, 0, pmag, pmag, 0]
    nmag = np.sqrt(nin[2]**2 + nin[3]**2 + nin[4]**2)
    n = [nin[0], nin[1], 0, 0, -nmag, nmag, 0]
    pdotn = dot4vec(p,n) # get p.n
    if len(EmissionVariables) == 0:
        return [p]
    qtildes_sq = [] # the evolution variables squared
    phis = [] # the generated phi angles
    pTs = [] # the pTs of the splittings
    msqs = [] # the virtualities squared
    
    # get the information generated in each emission
    for Emission in EmissionVariables:
        qtildesq = Emission[0] # the evolution variable t = qtilde^2
        z = Emission[1] # get the momentum fraction of the emission
        pT = Emission[2] # get the pT of the emission
        msq = Emission[3] # the virtuality of the emission
        phi = Emission[4] # get the phi of the emission
        
        # get the alphas for the quark and emitted gluon: 
        alpha = alpha_prime * (1-z) # alpha of emitted gluon
        alpha_prime *= z # alpha of evolving quark
        
        # append to lists:
        alphas.append(alpha) 
        alphas_prime.append(alpha_prime)
        qtildes_sq.append(qtildesq)
        phis.append(phi)
        pTs.append(pT)
        msqs.append(msq)
 
        # calculate the qT 4-vectors
        kT = [pT*np.cos(phi), pT*np.sin(phi), 0, 0] # (px, py, pz, E)
        qT = [0, 0, qT_prime[2]*(1-z) - kT[0], qT_prime[3]*(1-z) - kT[1], 0, 0] # (0, 0, px, py, pz, E)
        qT_prime =  [0, 0, qT_prime[2]*z + kT[0], qT_prime[3]*z + kT[1], 0, 0] # (0, 0, px, py, pz, E)
        qTs.append(qT)
        qTs_prime.append(qT_prime)
    msqs.append(0)
        
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
    
         

# define a unit vector given a vector
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

# get the angle between two 3-vectors
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# Given two 3d-vectors a, b find rotation of a so that its orientation matches b.
# Known as the Rodrigue's Rotation formula
# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
def GetRotationMatrixAB(a,b):
    acrossb = np.cross(a, b)
    #print(np.linalg.norm(acrossb))
    if np.linalg.norm(acrossb) > 1E-12:
        x = unit_vector(acrossb)
        theta = angle_between(a,b)
        I = np.identity(3)
        A = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
        R = I + A * np.sin(theta) + np.dot(A,A) * (1-np.cos(theta))
    else:
        return np.identity(3)
    return R

# Use the rotation to rotate the momenta to align with the mother particle's momentum in the lab frame
def RotateMomentaLab(p, Momenta):
    # array to return rotated momenta:
    RotatedMomenta = []
    # get the direction of the mother particle in a frame where it is aligned with the z-axis:
    pmag = np.sqrt(p[2]**2 + p[3]**2 + p[4]**2)
    pzonly = np.array([0,0,pmag])
    # get the rotation matrix
    Rmatrix = GetRotationMatrixAB(pzonly, np.array([p[2],p[3],p[4]]))
    # loop over the momenta and rotate them to the lab frame:
    for pm in Momenta:
        a = np.array([pm[2], pm[3], pm[4]])
        c = np.dot(Rmatrix,a)
        RotatedMomenta.append([pm[0], 1, c[0], c[1], c[2], pm[5], pm[6]]) 
    return RotatedMomenta

# sum up all the momenta and print out the resulting three-vector of total momentum
def CheckMomentumConservation(Momenta):
    totalmom = np.array([0,0,0])
    for pm in Momenta:
        if pm[1] == 1: # final-state only
            totalmom = totalmom + np.array([pm[2], pm[3], pm[4]])
    return totalmom

# implement global momentum conservation [see https://arxiv.org/pdf/0803.0883 section 6.4.2]
def GlobalMomCons(showeredParticles, showeredJets):
    # total energy:
    sqrthatS = 0
    # pj2 and qj2 arrays:
    pj2array = []
    qj2array = []
    # oldp and newq arrays:
    newqarray = []
    oldparray = []
    Rqparray = [] # the rotation angle array
    # loop over jets:
    for jet in showeredJets:
        #  get the 4-momentum of the progenitor jet
        oldp = np.array([jet[0][2], jet[0][3], jet[0][4], jet[0][5]])
        # Get the 3-momentum length of the "progenitor" of the jet (first entry in jet):
        pj2 = jet[0][2]**2 + jet[0][3]**2 +jet[0][4]**2
        sqrthatS += jet[0][5]
        # get the total momentum of the jet after showering
        qj = np.array([0,0,0,0])
        for p in jet[1]:
            qj = qj + np.array([p[2], p[3], p[4], p[5]])
            qj2 = qj[3]**2 - qj[0]**2 - qj[1]**2 - qj[2]**2
        if math.isnan(qj2):
            qj2 = 0
        Rqp = GetRotationMatrixAB(np.array([qj[0], qj[1], qj[2]]),np.array([oldp[0], oldp[1], oldp[2]]))
        pj2array.append(pj2)
        qj2array.append(qj2)
        newqarray.append(qj)
        oldparray.append(oldp)
        Rqparray.append(Rqp)
    # define the equation to get k
    def keqn(x):
        kres = 0
        for i in range(len(pj2array)):
            kres += np.sqrt(x * pj2array[i] + qj2array[i])
        kres = kres - sqrthatS
        return kres
    kres = np.sqrt(optimize.root(keqn, 0.99).x[0])
    if debug: print('kres=', kres)
    # now boost the momenta of the particles inside the jets according to the calculated kres:
    showeredParticlesBoosted = []
    # add back the initial state:
    for p in showeredParticles:
        if abs(p[0]) == 11:
            showeredParticlesBoosted.append(p)
    showeredJetsBoosted = []
    # check: if both parent particles have not radiated, put them in the record as they were:
    # if true, then at least one has radiated: 
    either_radiated = any(len(jet[1]) > 1 for jet in showeredJets)
    if either_radiated is False:
        for jj, jet in enumerate(showeredJets):
            showeredJetsBoosted.append(jet[1][0])
            showeredParticlesBoosted.append(jet[1][0])
    else:     
        for jj, jet in enumerate(showeredJets):
            showeredJetBoosted = []
            # get the boost:
            boostvec = getBoostBeta(kres, newqarray[jj], oldparray[jj])
            #if len(jet[1]) == 1: # don't do anything if the particle has not showered
            #    showeredJetsBoosted.append(jet[1][0])
            #    showeredParticlesBoosted.append(jet[1][0])
            # if the parents have radiated, rotate boost each particle in the jet for momentum conservation
            for p in jet[1]:
                # rotate all particles such that the new jet axis aligns with the parent jet axis
                protated = rotate(p, Rqparray[jj])
                #print('protated=', protated)
                # boost all particles as well:
                pboosted = boost(np.array([protated[2], protated[3], protated[4], protated[5]]), boostvec)
                #print('boostvec=', boostvec)
                #print('pboosted=', pboosted)
                # "id", "status", 'px [GeV]', 'py [GeV]', 'pz [GeV]', 'E [GeV]', 'm [GeV]']
                showeredParticlesBoosted.append([p[0], p[1], pboosted[0], pboosted[1], pboosted[2], pboosted[3], p[6]])
                showeredJetBoosted.append([p[0], p[1], pboosted[0], pboosted[1], pboosted[2], pboosted[3], p[6]])
            showeredJetsBoosted.append(showeredJetBoosted)
    # check:
    #testsum = 0
    #for jj, jet in enumerate(showeredJetsBoosted):
    #    sjet = np.array([0.,0.,0.,0.])
    #    for particle in jet:
    #        sjet += np.array([particle[2], particle[3], particle[4], particle[5]])
    #    testsum += sjet[3]
    #print('testsum=', testsum)
    return showeredParticlesBoosted


# gets the boost factor for global momentum conservation
# adapted from Herwig 7 Q-tilde shower
# newq is the 4-momentum of the outgoing jet (px, py, pz, E)
# oldp is the 4-momentum of the parent jet
def getBoostBeta(k, newq, oldp):
    qs = newq[0]**2 + newq[1]**2 + newq[2]**2
    q = np.sqrt(qs)
    Q2 = newq[3]**2 - newq[0]**2 - newq[1]**2 - newq[2]**2
    kp = k*np.sqrt(oldp[0]**2 + oldp[1]**2 + oldp[2]**2)
    kps = kp**2
    betam = (q*newq[3] - kp*np.sqrt(kps + Q2))/(kps + qs + Q2)
#  // usually we take the minus sign, since this boost will be smaller.
#  // we only require |k \vec p| = |\vec q'| which leaves the sign of
#  // the boost open but the 'minus' solution gives a smaller boost
#  // parameter, i.e. the result should be closest to the previous
#  // result. this is to be changed if we would get many momentum
#  // conservation violations at the end of the shower from a hard
#  // process.
#    betam = (q*np.sqrt(qs + Q2) - kp*np.sqrt(kps + Q2))/(kps + qs + Q2)
#  // move directly to 'return'
# NOTE: difference of a minus sign due to ThePEG's definition of the boost 
    beta = betam*(k/kp)*np.array([oldp[0], oldp[1], oldp[2]])
    #print('betam=', betam)
#  // note that (k/kp)*oldp.vect() = oldp.vect()/oldp.vect().mag() but cheaper. 
    if betam >= 0:
        return beta
    else:
        return np.array([0,0,0])

# boost in the direction of betavec
def boost(fourvector, betavec):
    if betavec.all() == 0:
        return fourvector
    # get the components of the boost vector
    betax = betavec[0]
    betay = betavec[1]
    betaz = betavec[2]
    beta = np.sqrt(betavec[0]**2 + betavec[1]**2 + betavec[2]**2)
    boosted = [0,0,0,0]
    gamma = 1./np.sqrt(1. - beta**2)
    boosted[3] = gamma * (fourvector[3] - betax * fourvector[0] - betay * fourvector[1] - betaz * fourvector[2])
    boosted[0] = - gamma * betax * fourvector[3] + (1 + (gamma-1)* betax**2 / beta**2) * fourvector[0] + (gamma-1) * betax * betay  * fourvector[1] / beta**2 + (gamma-1) * betax * betaz / beta**2 * fourvector[2]
    boosted[1] = - gamma * betay * fourvector[3] + (gamma - 1) * betay * betax  * fourvector[0]  / beta**2 + (1 + (gamma-1) * betay**2 / beta**2 ) * fourvector[1] + (gamma - 1) * betay * betaz * fourvector[2] / beta**2
    boosted[2] = - gamma * betaz * fourvector[3] + (gamma-1) * betaz * betax  * fourvector[0] / beta**2 + (gamma-1) * betaz * betay  * fourvector[1]/ beta**2 + (1 + (gamma-1) * betaz**2 / beta**2) * fourvector[2]
    return boosted


# rotate according to rotation matrix
def rotate(p, Rmatrix):
    # rotate
    a = np.array([p[2], p[3], p[4]])
    c = np.dot(Rmatrix,a)
    RotatedMomentum = np.array([p[0], 1, c[0], c[1], c[2], p[5], p[6]]) 
    return RotatedMomentum

# get the four-vector product for two vectors p and n
def dot4vec(p,n):
    return p[5] * n[5] - p[2] * n[2] - p[3] * n[3] - p[4] * n[4]


##########################
# Evolution begins here! #
##########################

# set the overestimate of alphaS once and for all:
alphaS_over = get_alphaS_over(Qc)
if debug: print('alphaS overestimate=', alphaS_over)

# read the LHE File
print('Showering', inputfile)
events, weights, multiweights = readlhefile(inputfile)

# Store the showered events:
showeredEvents = []

for i, particles in enumerate(tqdm(events)):
    # get the particles after parton shower and the showered jets: 
    showeredParticles, showeredJets = Shower(particles, pTmin, alphaS_over)
    # apply momentum conservation 
    showeredParticles = GlobalMomCons(showeredParticles, showeredJets)
    if debug is True or printevents is True:
        PrintMomenta(showeredParticles)
        print('Momentum conservation check AFTER=',CheckMomentumConservation(showeredParticles),'\n')
    showeredEvents.append(showeredParticles)
    if i > Nshower: break

# construct the HEPMC writer (Ascii)
#print('Writing output to HepMC file:', outputfile)
#hepmcwriter = pyhepmc.io.WriterAscii(outputfile)
# write hepmc file
#WriteHepMC(hepmcwriter, showeredEvents)

# construct the LHE writer:
sigma = 1.2
error = 0.2
ECM = 206
outlhe = outputfile.replace('.hepmc','_pyr.lhe')
fout = init_lhe(outlhe, sigma, error, ECM)
write_lhe(fout, showeredEvents, ECM**2, debug)
finalize_lhe(fout)
