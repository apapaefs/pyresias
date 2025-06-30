from random import * # random numbers
import os, subprocess # to check and create directories 
import math # python math
import numpy as np # numerical python
import scipy # scientific python
from scipy import optimize # for numerical solution of equations
from prettytable import PrettyTable # pretty printing of tables
from tqdm import tqdm # display progress
from optparse import OptionParser # command line parameters
from scipy.integrate import quad # for numerical integrals
from scipy import interpolate
from simplehisto import *
from alphaS import *

################################################
print('\nPyresias: a toy parton shower\n')
# a simple toy q -> q+g parton shower
#################################################

# function to print the emission information once they have been genrated:
def PrintEmissions(EmissionsArray):
    tbl = PrettyTable(["#", "Evo scale [GeV]", '1-z', 'pT [GeV]', 'virt. mass in a->bc [GeV]'])
    for i in range(len(EmissionsArray)):
        tbl.add_row([i, EmissionsArray[i][0], 1-EmissionsArray[i][1], EmissionsArray[i][2], EmissionsArray[i][3]])
    print(tbl)

#############
# SWITCHES: #
#############

# switch to print information or not:
debug = False
# switch to plot stuff or not:
plot = True

# output directory for plots
outputdirectory = 'plots/'

# RUN OPTIONS (defaults):

# the number of branches to evolve:
Nevolve = 100000
# the cutoff scale, e.g. 1 GeV. 
Qc = 1E-2
# the hard scale, e.g. 1000 GeV
Q = 1000.

# whether to impose kinematic limits on the generation of z if the Overstimate method is used for tMethod:
# this should be set to "False" if you are comparing with the splitting function
# to check if you are generating the correct distribution! 
# for a kinematically correct distributions, set to True!
Impose_zLimits = False

# Choose between solving for the evolution variable numerically ('Numerical') or through the overestimate ('Overestimated')
tMethod = 'Numerical' # 'Overestimated' or 'Numerical'

# maximum number of attempts (for the 'Numerical' option for tMethod)
ntry_max = 100
ntry = 0

##########################
# COMMAND LINE ARGUMENTS #
##########################

parser = OptionParser(usage="%prog [options]")

parser.add_option("-d", "--debug", dest="debug", default=False, action="store_true",
                  help="Print debugging to screen")

parser.add_option("-n", "--nevolve", dest="nevolve", default=Nevolve,
                  help="Set the number of evolution branches")

parser.add_option("-Q", dest="Q", default=Q,
                  help="Set the starting scale for the evolution")

parser.add_option("-c", dest="Qc", default=Qc,
                  help="Set the cutoff scale for the evolution")

parser.add_option("-o", dest="output", default=outputdirectory,
                  help="Set the output directory for plots")


# parse the command line arguments
(options, args) = parser.parse_args()

# set command line arguments 
debug = options.debug
Nevolve = int(options.nevolve)
Q = float(options.Q)
Qc = float(options.Qc)
outputdirectory = options.output + '/'

#################################################

# initialize alphaS: pass the value of alphaS at mz, and mz
aS = alphaS(0.118, 91.1876)

#################################################

# fixed scale if the alphaS is fixed:
scaleoption = "fixed" # "pt" for the default scale, "fixed" for a fixed scale, given by fixedScale:
fixedScale = Q/2. # if the choice above is "fixed"

if scaleoption == "fixed":
    print('WARNING: alphaS is fixed during the evolution to scale=', Q/2)
    print('Set scaleoption = "pt" if you wish to capture some higher-order corrections')

#################################################

# the q -> q + g splitting function
def Pqq(z): return CF * (1. + z**2)/(1.-z)
    
# the q -> q + g splitting function *overestimate* 
def Pqq_over(z): return 2.*CF/(1.-z)

# the scale choice of alphaS 
def scale_of_alphaS(t, z):
    if scaleoption == "pt":
        return z * (1-z) * np.sqrt(t)
    elif scaleoption == "fixed":
        return fixedScale

# return the true alphaS using the PDF alphaS over 2 pi
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
def get_alphaS_over(Q, Qcut):
    if scaleoption == "pt":
        scale = Qcut
    elif scaleoption == "fixed":
        scale = fixedScale
    alphaS_over = aS.alphasQ(scale)/2./np.pi
    if debug: print('alpha_S overestimate set to', alphaS_over, 'for scale=', scale, 'GeV')
    return alphaS_over

# get the momentum fraction candidate for the emission
def Get_zEmission(t, Qcut, R, aSover): return inversetGamma( tGamma(zm_over(t, Qcut), aSover) + R * ( tGamma(zp_over(t, Qcut), aSover) - tGamma(zm_over(t, Qcut), aSover)), aSover)
    
# calculate the transverse momentum of the emission
def Get_pTsq(t, z): return z**2 * (1-z)**2 * t

# calculate the virtual mass-squared of the emitting particle
def Get_mvirtsq(t,z): return z*(1-z) * t

# the function E(ln(t/Q**2)) = ln(t/Q**2) - (1/r) ln(R) for the numerical solution for the evolution scale, given random number R
def EmissionScaleFunc(logt_over_Qsq, Q, Qcut, R, aSover):
    # calculate t:
    t = Q**2 * np.exp( logt_over_Qsq )
    # get r:
    r = tGamma(zp_over(t, Qcut), aSover) - tGamma(zm_over(t, Qcut), aSover)
    # calculate E(ln(t/Q**2)), the equation to solve
    return logt_over_Qsq - (1./r) * np.log(R)

# a function that calculates (numerically) the emission scale given the initial scale Q, cutoff Qc and random number R
def Get_tEmission(Q, Qcut, R, aSover):
    global ntry
    tfac = 4-1E-10
    tolerance = 1E-3 # the tolerance for the solution
    popt = [Q, Qcut, R, aSover] # options to pass to the function for the solver
    EmissionFunc_arg = lambda x : EmissionScaleFunc(x, *popt) # the function in a form appropriate for the solver
    # calculate the solution using "Ridder's" method
    sol, results = scipy.optimize.ridder(EmissionFunc_arg, np.log(tfac*Qcut**2/Q**2), 0., xtol=tolerance, full_output=True, maxiter=1000)
    # get the actual evolution variable from the solution
    tEm_sol = Q**2 * np.exp( sol )
    if math.isnan(tEm_sol) or tEm_sol < 4*Qcut**2:
        if debug: print('\tEmission fails due to NaN tEm or tEm_sol < 4*Qcut**2, tEm_sol=', tEm_sol, 'R1=', R)
        return Q**2, [], False, False
    # if a solution has not been found, terminate the evolution
    if debug: print('\t\tabs(EmissionFunc_arg(sol))=', abs(EmissionFunc_arg(sol)), 'ntry=', ntry)
    if ntry > ntry_max:
        ntry = 0
        return Q**2, [], False, False
    if abs(EmissionFunc_arg(sol)) > tolerance:
        ntry += 1
        return Q**2, [], True, False
    # otherwise return the emission scale and continue
    return tEm_sol, results, True, True

# a function that calculates the emission scale given the initial scale Q, cutoff Qc and random number R
def Get_tEmission_direct(Q, Qcut, R, aSover):
    generated = True
    upper = tGamma(zp_over(Q**2, Qcut), aSover)
    lower = tGamma(zm_over(Q**2, Qcut), aSover)
    if lower > upper:
        if debug: print('\tEmission fails due upper < lower')
        return Q**2, [], False
    c = 1/(upper - lower)
    # get the actual evolution variable
    tEm = Q**2 * R**c
    if math.isnan(tEm) or tEm < 4*Qcut**2:
        if debug: print('\tEmission fails due to NaN tEm or tEm_sol < 4*Qcut**2, tEm_sol=', tEm, 'R1=', R)
        return Q**2, [], False, False
    return tEm, [], True, True

# function that generates emissions:
def Generate_Emission(Q, Qcut, aSover):
    generated = True
    # generate random numbers
    R1 = random()
    R2 = random()
    R3 = random()
    R4 = random()
    # solve for the (candidate) emission scale:
    if tMethod == 'Overestimated':
        tEm, results, continueEvolution, generated = Get_tEmission_direct(Q, Qcut, R1, aSover)
    elif tMethod == 'Numerical':
        tEm, results, continueEvolution, generated = Get_tEmission(Q, Qcut, R1, aSover)
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
    if tMethod == 'Numerical':
        zEm = Get_zEmission(tEm**2, Qcut, R2, aSover)
    elif tMethod == 'Overestimated':
        zEm = Get_zEmission(Q**2, Qcut, R2, aSover)
    if debug: print('\t\tcandidate momentum fraction, zEm=', zEm)
    # check that zEm is within allowed limits:
    # this needs to be DISABLED for comparison with the full splitting function!
    if tMethod == 'Overestimated' and Impose_zLimits is True:
        if zEm < zm_true or zEm > zp_true:
            if debug: print('\t\temission REJECTED due to zEm < zm_true or zEm > zp_true: zEm=', zEm, 'zm_true', zm_true, 'zp=', zp_true)
            generated = False
    # get the transverse momentum 
    pTsqEm = Get_pTsq(tEm, zEm)
    if debug: print('\t\tcandidate transverse momentum =', np.sqrt(pTsqEm))
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
        # NOTE: tEm continues from the rejected emission scale!
    # return all the variables for the emission
    return tEm, zEm, pTsqEm, MsqEm, generated, continueEvolution

# the function that performs the evolution of a single branch:
# returns a list of all the emissions for further processing
def Evolve(Q, Qmin, aSover):
    # the minimum evolution scale
    tEm_min = Qmin**2
    # counter for the number of emissions:
    Nem = 0
    # array to store emission info:
    Emissions = []
    fac_cutoff = 4. # actual cutoff = fac_cutoff * Qc**2
    # star the evolution
    tEm = Q**2 # initial value of the evolution variable
    zEm = 1 # initial value of the momentum fraction
    if debug: print('generating evolution for Q=', Q, 'GeV\n')
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
            if debug: print('total number of emissions=', Nem)
            return Emissions
        # if the emission was successful, append to the Emissions list and continue
        if zEm != 1:
            Emissions.append([np.sqrt(tEm), zEm, np.sqrt(pTsqEm), np.sqrt(MsqEm)])
            if debug: print('\t->successful emission at sqrt(t)=', np.sqrt(tEm), 'z=', zEm, 'pT=', np.sqrt(pTsqEm), 'mVirt=', np.sqrt(MsqEm))
            Nem = Nem + 1
    if debug:
        print('no further emissions, evolution ended')
        print('total number of emissions=', Nem)
        print('\n')
        print('-----')
        print('Emissions table:')
        PrintEmissions(Emissions)
    return Emissions

##########################
# Evolution begins here! #
##########################

# set the overestimate of alphaS once and for all:
alphaS_over = get_alphaS_over(Q, Qc)
if debug: print('alphaS overestimate=', alphaS_over)
    
# list to store all emission information:
AllEmissions = []

print('Evolving', Nevolve, 'branches from Q=', Q, 'GeV --> Qc=', Qc, 'GeV')

# perform evolution over Nevolve branches:
for j in tqdm(list(range(Nevolve))): # tqdm is the progress bar 
    # perform the evolution over a single branch j and return a list of all the emissions:
    Emissions = Evolve(Q,Qc,alphaS_over)
    # concatenate the emissions of this evolution branch to the list of all emissions to plot
    AllEmissions = AllEmissions + Emissions

print('All evolutions ended')

##############################
# Evolution has ended, plot! #
##############################

if plot == False:
    print('not plotting anything! exiting...')
    exit()

# create output directory if it does not exist:
if os.path.exists(outputdirectory) == False:
    mkdircommand = 'mkdir ' + outputdirectory
    p = subprocess.Popen(mkdircommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd='.')
    
############


# rotate emission array:
AllEmissionsPlot = np.array(AllEmissions).T

# evolution variable:
tarray = AllEmissionsPlot[0]
simplehisto('plotting evolution variable emission scales', 'evolutionvar', outputdirectory, tarray, r'$P(\sqrt{\tilde{t}})$', r'$\sqrt{\tilde{t}}$ [GeV]')

# transverse momentum:
ptarray = AllEmissionsPlot[2]
simplehisto('plotting pT of emissions', 'transversemom', outputdirectory, ptarray, '$P(p_T)$', '$p_T$ [GeV]')

# virtual mass:
marray = AllEmissionsPlot[3]
simplehisto('plotting virtual mass of emissions', 'virtmass', outputdirectory, marray, '$P(m_\\mathrm{virt}(a\\rightarrow bc))$', '$m_\\mathrm{virt}(a\\rightarrow bc)$ [GeV]')


###########################
###########################
###########################
nbins=50
print('---')
print('plotting z of emissions (1-z)*P(z)')
# plot settings ########
plot_type = 'momentumfrac'
# plot:
# plot settings
ylab = '$(1-z)P(z)$'
xlab = '$z$'
ylog = False
xlog = False
nbins=40
# construct the axes for the plot
fig = plt.figure(constrained_layout=True)
fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0,
                            wspace=0)
gs = gridspec.GridSpec(4, 4,figure=fig,wspace=0, hspace=0)
ax = fig.add_subplot(gs[:3, :])
ax2 = fig.add_subplot(gs[3, :])
gs.update(wspace=0,hspace=0)

ax.grid(False)
ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
ax2.yaxis.set_minor_locator(MultipleLocator(0.025))

tarray = []
for i in range(len(AllEmissions)):
    tarray.append(np.array(AllEmissions[i][1]))
gs.update(wspace=0.0, hspace=0.0)

# get the histogram bins:
bins, edges = np.histogram(tarray, bins=nbins, density=True)
left,right = edges[:-1],edges[1:]
X = np.array([0.5*left+0.5*right]).T.flatten()
Y = np.array([bins]).T.flatten() * (1-X)
# normalise:
xnorm_min=0.0
xnorm_max=1.0

#Y = Y/np.linalg.norm(Y)/(Y[1]-Y[0])
#Ysum = Y[(X>xnorm_min) & (X<xnorm_max)].sum()
gs.update(wspace=0.0, hspace=0.0)


# compare to the input splitting function
# this comparison is only correct if alphaS is fixed
# this is because the scale of alphaS is also a function of z 
Yspl = Pqq(X) * (1-X)

# get the integral numerically, but not in the whole range
# since the splitting function diverges as z->1 and this cannot be captured numerically:
zp = X[(X<xnorm_max)][-1]
zm = X[(X>xnorm_min)][0]
def Pqq1mz(z):
    return Pqq(z) * (1-z)
YsplI = quad(Pqq1mz, 0, 1)
YsplI = np.linalg.norm(Yspl)
print(np.linalg.norm(Yspl))
ax.plot(X,Yspl, color='blue', lw=1, label='Splitting function', marker='x', ms=0)

# plot:
print(np.linalg.norm(Y))
Y = YsplI * Y / np.linalg.norm(Y)
print(np.linalg.norm(Y))
print(YsplI)

ax.plot(X,Y, label='Pyresias', color='red', lw=0, marker='o', ms=2)


# ratio:
ax2.plot(X,Y/Yspl, color='red', lw=0, label='Splitting function', marker='o', ms=2)
ax2.hlines(y=1, xmin=0, xmax=1, color='black', ls='--')

# set the ticks, labels and limits etc.
ax.set_ylabel(ylab, fontsize=20)
ax2.set_xlabel(xlab, fontsize=20)
ax2.set_ylabel('Pyr./Spl.')
ax2.set_ylim(0.9,1.1)
ax2.set_xlim(0.0,1.0)
ax.set_xlim(0.0,1.0)
# choose x and y log scales
if ylog:
    ax.set_yscale('log')
else:
    ax.set_yscale('linear')
if xlog:
    ax.set_xscale('log')
else:
    ax.set_xscale('linear')

# create legend and plot/font size
ax.legend()
ax.legend(loc="upper center", numpoints=1, frameon=False, prop={'size':10})
ax.set_xticklabels('')
ax.set_xticks([])

# save the figure
print('saving the figure')
# save the figure in PDF format
infile = plot_type + '.dat'
print('---')
print('output in', outputdirectory + infile.replace('.dat','.pdf'))
plt.savefig(outputdirectory + infile.replace('.dat','.pdf'), bbox_inches='tight')
plt.close(fig)

###########################
###########################
###########################
nbins=50
print('---')
print('plotting z of emissions P(z)')
# plot settings ########
plot_type = 'momentumfrac_p'
# plot:
# plot settings
ylab = '$P(z)$'
xlab = '$z$'
ylog = True
xlog = False
nbins=40
# construct the axes for the plot
fig = plt.figure(constrained_layout=True)
fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0,
                            wspace=0)
gs = gridspec.GridSpec(4, 4,figure=fig,wspace=0, hspace=0)
ax = fig.add_subplot(gs[:3, :])
ax2 = fig.add_subplot(gs[3, :])
gs.update(wspace=0,hspace=0)

ax.grid(False)
ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
ax2.yaxis.set_minor_locator(MultipleLocator(0.025))

tarray = []
for i in range(len(AllEmissions)):
    tarray.append(np.array(AllEmissions[i][1]))
gs.update(wspace=0.0, hspace=0.0)

# get the histogram bins:
bins, edges = np.histogram(tarray, bins=nbins, density=True)
left,right = edges[:-1],edges[1:]
X = np.array([0.5*left+0.5*right]).T.flatten()
Y = np.array([bins]).T.flatten() 
# normalise:
xnorm_min=0.0
xnorm_max=1.0

#Y = Y/np.linalg.norm(Y)/(Y[1]-Y[0])
#Ysum = Y[(X>xnorm_min) & (X<xnorm_max)].sum()
gs.update(wspace=0.0, hspace=0.0)


# compare to the input splitting function
# this comparison is only correct if alphaS is fixed
# this is because the scale of alphaS is also a function of z 
Yspl = Pqq(X) 

# get the integral numerically, but not in the whole range
# since the splitting function diverges as z->1 and this cannot be captured numerically:
zp = X[(X<xnorm_max)][-1]
zm = X[(X>xnorm_min)][0]
YsplI = quad(Pqq, 0, 1-1E-10)
YsplI = np.linalg.norm(Yspl)
print(np.linalg.norm(Yspl))
ax.plot(X,Yspl, color='blue', lw=1, label='Splitting function', marker='x', ms=0)

# plot:
print(np.linalg.norm(Y))
Y = YsplI * Y / np.linalg.norm(Y)
print(np.linalg.norm(Y))
print(YsplI)

ax.plot(X,Y, label='Pyresias', color='red', lw=0, marker='o', ms=2)


# ratio:
ax2.plot(X,Y/Yspl, color='red', lw=0, label='Splitting function', marker='o', ms=2)
ax2.hlines(y=1, xmin=0, xmax=1, color='black', ls='--')

# set the ticks, labels and limits etc.
ax.set_ylabel(ylab, fontsize=20)
ax2.set_xlabel(xlab, fontsize=20)
ax2.set_ylabel('Pyr./Spl.')
ax2.set_ylim(0.9,1.1)
ax2.set_xlim(0.0,1.0)
ax.set_xlim(0.0,1.0)
# choose x and y log scales
if ylog:
    ax.set_yscale('log')
else:
    ax.set_yscale('linear')
if xlog:
    ax.set_xscale('log')
else:
    ax.set_xscale('linear')

# create legend and plot/font size
ax.legend()
ax.legend(loc="upper center", numpoints=1, frameon=False, prop={'size':10})
ax.set_xticklabels('')
ax.set_xticks([])

# save the figure
print('saving the figure')
# save the figure in PDF format
infile = plot_type + '.dat'
print('---')
print('output in', outputdirectory + infile.replace('.dat','.pdf'))
plt.savefig(outputdirectory + infile.replace('.dat','.pdf'), bbox_inches='tight')
plt.close(fig)


print('\nDone!')
