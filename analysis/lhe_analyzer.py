import sys
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math
from tqdm import tqdm
import matplotlib.gridspec as gridspec # more plotting 
import gzip
import fastjet
import awkward as ak 
from matplotlib import ticker 
###########################################################
# Handle the input here. 
# The default output file tag (i.e. extension) is 'output'.
###########################################################

if len(sys.argv) < 2:
    print('lhe_analyzer.py [lhe file] ([output file tag])')
    exit()
    
inputfile = str(sys.argv[1])

outputfiletag = ''

if len(sys.argv) > 2:
    inputfile2 = str(sys.argv[2])

if len(sys.argv) > 3:
    inputfile3 = str(sys.argv[3])

if len(sys.argv) > 4:
    inputfile4 = str(sys.argv[4])
     

##########################
# FUNCTIONS
##########################

# Functions to handle colors in matplotlib plots:
# choose the next colour -- for plotting
ccount = 0
def next_color():
    global ccount
    colors = ['green', 'red', 'orange', 'blue', 'black', 'cyan', 'magenta', 'brown', 'violet'] # 9 colours
    color_chosen = colors[ccount]
    if ccount < 8:
        ccount = ccount + 1
    else:
        ccount = 0    
    return color_chosen
# do not increment colour in this case:
def same_color():
    global ccount
    colors = ['green', 'red', 'orange', 'blue', 'black', 'cyan', 'magenta', 'brown', 'violet'] # 9 colours
    color_chosen = colors[ccount-1]
    return color_chosen
# reset the color counter:
def reset_color():
    global ccount
    ccount = 0


# function to plot histograms: including the cross section normalization and STACKED 
# DATA_array contains ARRAYS of data for each event. Each array represents a different type of input (e.g. a run with different parameters, etc.).
# CrossSection_array contains ARRAYS of cross sections 
# plot_type is simply the main name of the plot
# plotnames_multi has to be an array of equal size to DATA_array
# custom_bins can be provided for the desired observable
def histogram_multi_xsec_stacked(DATA_array, CrossSection_array, plot_type, plotnames_multi, xlabel='', ylabel='fraction/bin', nbins=50, title='', custom_bins=[], ylogbool=False, xlogbool=False):
    print('---')
    print('plotting', plot_type)
    
    # plot settings ########
    ylab = ylabel # the ylabel
    xlab = xlabel # the x label
    # log scale?
    ylog = ylogbool # whether to plot y in log scale
    xlog = xlogbool # whether to plot x in log scale

    # construct the axes for the plot
    fig = plt.figure(constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0,
                            wspace=0)
    gs = gridspec.GridSpec(4, 4,figure=fig,wspace=0, hspace=0)
    ax = fig.add_subplot(gs[:3, :])
    ax2 = fig.add_subplot(gs[3, :])
    gs.update(wspace=0,hspace=0)

    # loop over the DATA in the DATA_array
    # get the errors per bin and normalize so that we obtain fraction of events/bin
    dd = 0
    X = []
    Y = []
    binstot = [] 
    for DATA in DATA_array:
        if len(custom_bins) == 0:
            bins, edges = np.histogram(np.array(DATA), bins=nbins)
        else:
            bins, edges = np.histogram(np.array(DATA), bins=custom_bins)
        errors = np.divide(np.sqrt(bins), bins, out=np.zeros_like(np.sqrt(bins)), where=bins!=0.)
        bins = bins/float(len(DATA))
        errors = bins*errors
        #print(bins)
        #print(errors)
        left,right = edges[:-1],edges[1:]
        X = np.array([left,right]).T.flatten()
        if len(Y) == 0:
            Y = np.array([bins,bins]).T.flatten()*CrossSection_array[dd]
            binstot = bins/float(len(DATA))*CrossSection_array[dd]
        else: 
            Y = Y + np.array([bins,bins]).T.flatten()*CrossSection_array[dd]
            binstot = binstot + bins/float(len(DATA))*CrossSection_array[dd]
        dd = dd+1
    center = (edges[:-1] + edges[1:]) / 2
    plt.plot(X,Y, label='total', color=next_color(), lw=1)
    #plt.errorbar(X, Y, yerr=., color=same_color(), lw=0, elinewidth=1, capsize=1)


    # set the ticks, labels and limits etc.
    ax.set_ylabel(ylab, fontsize=20)
    ax2.set_ylabel('Pyr./HW7')
    ax2.set_ylim(0.9,1.1)
    ax2.set_xlabel(xlab, fontsize=20)
    
    # choose x and y log scales
    if ylog:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
    if xlog:
        ax.set_xscale('log')
    else:
        ax.set_xscale('linear')

    # set the limits on the x and y axes if required below:
    # (this is not implemented automatically yet)
    #xmin = 0.
    #xmax = 1500.
    #ymin = 0.
    #ymax = 0.09
    #plt.xlim([0,400])
    #plt.ylim([0.06,0.15])

    # create legend and plot/font size
    ax.legend()
    ax.legend(loc="upper right", numpoints=1, frameon=False, prop={'size':8})
    
    # set the title of the figure
    if title != '':
        plt.title(title)
    
    # save the figure
    print('saving the figure')
    # save the figure in PDF format
    if outputfiletag != '':
        infile = plot_type + '-' + outputfiletag + '_stacked.dat'
    else:
        infile = plot_type + '_stacked.dat'
    print('output in', infile.replace('.dat','.pdf'))
    plt.savefig(infile.replace('.dat','.pdf'), bbox_inches='tight')
    plt.close(fig)


# function to plot histograms: including the cross section normalization
# CrossSection_array contains ARRAYS of cross sections 
# DATA_array contains ARRAYS of data for each event. Each array represents a different type of input (e.g. a run with different parameters, etc.). 
# plot_type is simply the main name of the plot
# plotnames_multi has to be an array of equal size to DATA_array
# custom_bins can be provided for the desired observable
def histogram_multi_xsec(DATA_array, CrossSection_array, plot_type, plotnames_multi, xlabel='', ylabel='fraction/bin', nbins=50, title='', custom_bins=[], ylogbool=False, xlogbool=False):
    print('---')
    print('plotting', plot_type)
    reset_color()
    # plot settings ########
    ylab = ylabel # the ylabel
    xlab = xlabel # the x label
    # log scale?
    ylog = ylogbool # whether to plot y in log scale
    xlog = xlogbool # whether to plot x in log scale

    # construct the axes for the plot
    fig = plt.figure(constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0,
                            wspace=0)
    gs = gridspec.GridSpec(4, 4,figure=fig,wspace=0, hspace=0)
    ax = fig.add_subplot(gs[:3, :])
    ax2 = fig.add_subplot(gs[3, :])
    gs.update(wspace=0,hspace=0)
    
    # loop over the DATA in the DATA_array
    # get the errors per bin and normalize so that we obtain fraction of events/bin
    dd = 0
    Yarray = []
    for DATA in DATA_array:
        if len(custom_bins) == 0:
            bins, edges = np.histogram(np.array(DATA), bins=nbins)
        else:
            bins, edges = np.histogram(np.array(DATA), bins=custom_bins)
        errors = np.divide(np.sqrt(bins), bins, out=np.zeros_like(np.sqrt(bins)), where=bins!=0.)
        bins = bins/float(len(DATA))
        errors = bins*errors
        #print(bins)
        #print(errors)
        left,right = edges[:-1],edges[1:]
        X = np.array([left,right]).T.flatten()
        Y = np.array([bins,bins]).T.flatten()
        Xc = np.array([0.5*(left+right)]).T.flatten()
        Yc = np.array([bins]).T.flatten()

        Yc = Yc/np.linalg.norm(Yc)*CrossSection_array[dd]
        Yarray.append(np.array(Yc))
        
        ax.plot(X,Y, label=plotnames_multi[dd], color=next_color(), lw=1)
        #center = (edges[:-1] + edges[1:]) / 2
        #plt.errorbar(center, bins, yerr=errors, color=same_color(), lw=0, elinewidth=1, capsize=1)
        dd = dd+1
    
    # set the ticks, labels and limits etc.
    ax.set_ylabel(ylab, fontsize=20)
    ax2.set_ylabel('Pyr./HW7')
    ax2.set_ylim(0.5,2.0)
    ax2.set_xlabel(xlab, fontsize=20)
    ax.set_xlim([X[0], X[-1]])
    ax2.set_xlim([X[0], X[-1]])

    ax2.plot(Xc,Yarray[1]/Yarray[0], color='red', marker='o', markersize=3, lw=0)
    ax2.hlines(y=1, xmin=X[0], xmax=X[-1], color='black', ls='--')
    # choose x and y log scales
    if ylog:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
    if xlog:
        ax.set_xscale('log')
    else:
        ax.set_xscale('linear')


    # set the x axis ticks automatically
    ax.tick_params(labelbottom=False)
    ax2.xaxis.set_major_locator(ticker.AutoLocator())
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.yaxis.set_major_locator(ticker.AutoLocator())
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # legend
    ax.legend()
    ax.legend(loc="upper right", numpoints=1, frameon=False, prop={'size':8})
    ax.set_xticklabels('')
    ax.set_xticks([])

    
    # set the title of the figure
    if title != '':
        ax.set_title(title)
    
    # save the figure
    print('saving the figure')
    # save the figure in PDF format
    if outputfiletag != '':
        infile = plot_type + '-' + outputfiletag + '.dat'
    else:
        infile = plot_type + '.dat'
    print('output in', infile.replace('.dat','.pdf'))
    plt.savefig(infile.replace('.dat','.pdf'), bbox_inches='tight')
    plt.close(fig)

    
# function to plot histograms
# DATA_array contains ARRAYS of data for each event. Each array represents a different type of input (e.g. a run with different parameters, etc.). 
# plot_type is simply the main name of the plot
# plotnames_multi has to be an array of equal size to DATA_array
# custom_bins can be provided for the desired observable
def histogram_multi(DATA_array, plot_type, plotnames_multi, xlabel='', ylabel='fraction/bin', nbins=50, title='', custom_bins=[], ylogbool=False, xlogbool=False):
    print('---')
    print('plotting', plot_type)
    
    # plot settings ########
    ylab = ylabel # the ylabel
    xlab = xlabel # the x label
    # log scale?
    ylog = ylogbool # whether to plot y in log scale
    xlog = xlogbool # whether to plot x in log scale

    # construct the axes for the plot
    # no need to modify this if you just need one plot
    gs = gridspec.GridSpec(4, 4)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(False)

    # loop over the DATA in the DATA_array
    # get the errors per bin and normalize so that we obtain fraction of events/bin
    dd = 0
    for DATA in DATA_array:
        if len(custom_bins) == 0:
            bins, edges = np.histogram(np.array(DATA), bins=nbins)
        else:
            bins, edges = np.histogram(np.array(DATA), bins=custom_bins)
        errors = np.divide(np.sqrt(bins), bins, out=np.zeros_like(np.sqrt(bins)), where=bins!=0.)
        bins = bins/float(len(DATA))
        errors = bins*errors
        #print(bins)
        #print(errors)
        left,right = edges[:-1],edges[1:]
        X = np.array([left,right]).T.flatten()
        Y = np.array([bins,bins]).T.flatten()
        plt.plot(X,Y, label=plotnames_multi[dd], color=next_color(), lw=1)
        center = (edges[:-1] + edges[1:]) / 2
        plt.errorbar(center, bins, yerr=errors, color=same_color(), lw=0, elinewidth=1, capsize=1)
        dd = dd+1
    

    # set the ticks, labels and limits etc.
    ax.set_ylabel(ylab, fontsize=20)
    ax.set_xlabel(xlab, fontsize=20)
    
    # choose x and y log scales
    if ylog:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
    if xlog:
        ax.set_xscale('log')
    else:
        ax.set_xscale('linear')

    # set the limits on the x and y axes if required below:
    # (this is not implemented automatically yet)
    #xmin = 0.
    #xmax = 1500.
    #ymin = 0.
    #ymax = 0.09
    #plt.xlim([0,400])
    #plt.ylim([0.06,0.15])

    # create legend and plot/font size
    ax.legend()
    ax.legend(loc="upper right", numpoints=1, frameon=False, prop={'size':8})
    
    # set the title of the figure
    if title != '':
        plt.title(title)
    
    # save the figure
    print('saving the figure')
    # save the figure in PDF format
    if outputfiletag != '':
        infile = plot_type + '-' + outputfiletag + '.dat'
    else:
        infile = plot_type + '.dat'
    print('output in', infile.replace('.dat','.pdf'))
    plt.savefig(infile.replace('.dat','.pdf'), bbox_inches='tight')
    plt.close(fig)

# function to read lhe files in and grab the particle momenta for each event
# it also grabs the weight of each event, as well as the multiple weights, if present
# the return variables are: events, in which each entry is a set of particle 4-momenta in the format:
# [id, status, px, py, pz, e, m] -> id is the PDG id, status is the LHE status (i.e. incoming: -1, final: 1)
# weights contains the weight of each event
# multiweights contains the multiple weights of each event 
def readlhefile(infile):
    if infile.endswith('.gz'):
        my_open = gzip.open
    else:
        my_open = open
    infile_read = my_open(infile, 'rt')
    numevents = 0
    reading_event = False
    events = []
    weights = []
    multiweights = []
    for line in infile_read:
        if '<event>' in line:
            particles = []
            multiweight = {}
            #print('reading new event')
            numevents = numevents + 1
            reading_event = True
        if reading_event is True:
            if '</event>' in line:
                reading_event = False
                events.append(particles)
                weights.append(weight)
                multiweights.append(multiweight)
            #print(line, len(line.split()))
            if len(line.split()) == 6:
                weight = float(line.split()[2])
            if len(line.split()) == 13:
                particles.append(read_momenta(line))
            if len(line.split()) == 4:
                multiweight[line.split()[1].replace('id=', '').replace('>', '').replace("'", '')] = float(line.split()[2])
                #print('multiweight[', line.split()[1].replace('id=', '').replace('>', '').replace("'", ''), ']=', line.split()[2])
                
    return events, weights, multiweights

# read the particle information for the given particle line in the LHE file
def read_momenta(inputline):
    id = int(inputline.split()[0])
    status = int(inputline.split()[1])
    px = float(inputline.split()[6])
    py = float(inputline.split()[7])
    pz = float(inputline.split()[8])
    e = float(inputline.split()[9])
    m = float(inputline.split()[10])
    return [id, status, px, py, pz, e, m]


# convert to fastjet, but only if the pt is > minptc
def convert_tofj(momin, minptc, maxrapc):
    arrayout = []
    for mm in range(len(momin)):
        #print(momin[mm][1], momin[mm][2], momin[mm][3], momin[mm][0])
        fj = fastjet.PseudoJet(momin[mm][1], momin[mm][2], momin[mm][3], momin[mm][0])
        if fj.perp() > minptc and abs(fj.eta()) < maxrapc:
            arrayout.append(fj)
    return arrayout


# calculate the thrust given a set of *3-momenta*
def get_thrust(momenta):
    ns = 10
    phi = np.linspace(0, 2*np.pi, ns)
    theta = np.linspace(0, np.pi, ns)
    Tvals = []
    for ph in phi:
        for th in theta:
            Tval = 0
            n = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
            for p in momenta:
                p = np.array(p)
                Tval += abs(np.dot(p,n))
                Tvals.append(Tval)
    denom = 0
    for p in momenta:
        denom += abs(np.linalg.norm(p))
    Tmax = np.max(Tvals)/denom
    return Tmax
                
            
        

############################################################
# Define your ANALYSIS function here!
# The example looks for the new particle with pdg id "99925"
# and calculates its transverse momentum.
# For each observable we wish to return, we must add it to
# the dictionary "output_dictionary" as in the example below.
#############################################################
def analyze(events, weights):
    # a dictionary that contains the arrays that we wish to plot
    output_dictionary = {}
    # construct the observables by putting emtpy arrays into the dictionary:
    output_dictionary['costheta'] = []
    output_dictionary['pt'] = []
    output_dictionary['ptg'] = []
    output_dictionary['njets'] = []
    output_dictionary['ptj1'] = []
    output_dictionary['ptj2'] = []
    output_dictionary['ptj'] = []
    output_dictionary['sumpvecmag'] = []
    output_dictionary['thrust'] = []
    output_dictionary['yg'] = []
    output_dictionary['Eq'] = []
    output_dictionary['Eg'] = []
    output_dictionary['pzq'] = []
    output_dictionary['pzg'] = []
    output_dictionary['ng'] = []
    output_dictionary['minvg'] = []

    # fastjet:
    jetalgo = fastjet.antikt_algorithm
    jetR = 0.4
    jetdef = fastjet.JetDefinition(jetalgo, jetR)
    minpt = 0.0
    # loop over the particles in the event:
    for iev, particles in tqdm(enumerate(events)):
        sumpvec = np.array([0.,0.,0.])
        momtocluster = []
        momvec = []
        ng = 0 # count the number of emitted gluons
        for p in particles:
            if p[1] == 1:
                #print('p[2], p[3], p[4]=',p[2], p[3], p[4])
                sumpvec += np.array([p[2], p[3], p[4]])
            if (abs(p[0])>0 and abs(p[0])<6) or p[0] == 21:
                momtocluster.append([p[2], p[3], p[4], p[5]])
                momvec.append(np.array([p[2], p[3], p[4]]))
            if p[0] == 21:
                ng += 1
                pt = math.sqrt(p[2]**2 + p[3]**2)
                output_dictionary['ptg'].append(pt)
                y = 0.5 * np.log( (p[5] + p[4])/(p[5] - p[4]) )
                output_dictionary['yg'].append(y)
                pz = p[4]
                E = p[5]
                minvg = p[5]**2 - (p[2]**2 + p[3]**2 + p[4]**2)
                output_dictionary['minvg'].append(minvg)
                output_dictionary['pzg'].append(pz)
                output_dictionary['Eg'].append(E)
            if abs(p[0])>0 and abs(p[0])<6:
                costheta = p[4]/math.sqrt(p[2]**2 + p[3]**2 + p[4]**2)
                output_dictionary['costheta'].append(costheta)
                pt = math.sqrt(p[2]**2 + p[3]**2)
                E = p[5]
                pz = p[4]
                output_dictionary['Eq'].append(E)
                output_dictionary['pt'].append(pt)
                output_dictionary['pzq'].append(pz)
        output_dictionary['ng'].append(ng)
        #Thrust = get_thrust(momvec)
        #output_dictionary['thrust'].append(Thrust)
        sumpvecmag = math.sqrt(sumpvec[0]**2 + sumpvec[1]**2 + sumpvec[2]**2)
        output_dictionary['sumpvecmag'].append(sumpvecmag)
        momfj = convert_tofj(momtocluster, 0,100)
        cluster = fastjet.ClusterSequence(momfj, jetdef)
        jets = fastjet.sorted_by_pt(cluster.inclusive_jets(minpt))
        output_dictionary['njets'].append(len(jets))
        for j in jets:
            output_dictionary['ptj'].append(j.perp())
        output_dictionary['ptj1'].append(jets[0].perp())
        if len(jets) > 1:
            output_dictionary['ptj2'].append(jets[1].perp())
    return output_dictionary




#######################################
# PERFORM THE ANALYSIS AND PLOT HERE:
#######################################

# read the LHE File
print('Reading', inputfile)
events, weights, multiweights = readlhefile(inputfile)
# analyze the events by passing them to the analysis fuinction defined above
output = analyze(events, weights)

if len(sys.argv) > 2:
    print('Reading', inputfile2)
    events2, weights2, multiweights2 = readlhefile(inputfile2)
    # analyze the events by passing them to the analysis fuinction defined above
    output2 = analyze(events2, weights2)

if len(sys.argv) > 3:
    print('Reading', inputfile3)
    events3, weights3, multiweights3 = readlhefile(inputfile3)
    # analyze the events by passing them to the analysis fuinction defined above
    output3 = analyze(events3, weights3)

if len(sys.argv) > 4:
    print('Reading', inputfile4)
    events4, weights4, multiweights4 = readlhefile(inputfile4)
    # analyze the events by passing them to the analysis fuinction defined above
    output4 = analyze(events4, weights4)
    
CrossSections = [1, 1]
# plot all the variables in the output dictionary. 
# here as an example we are plotting the heavy scalar pT.
# Note that "histogram_multi" takes as input in DATA_array an array of data points,
# hence the extra [] there and in the plotnames_multi
if len(sys.argv) == 3:
    histogram_multi_xsec([output['pt'], output2['pt']], [1.0, 1.0], 'pt', [r'HERWIG 7', r'Pyresias'], xlabel=r'$p_T$ of outgoing quarks [GeV]', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} p_T}$ [GeV$^{-1}$]', custom_bins=np.arange(0,120, 5))
    histogram_multi_xsec([output['ptg'], output2['ptg']], [1.0, 1.0], 'ptg', [r'HERWIG 7', r'Pyresias'], xlabel=r'$p_T$ of emitted gluons [GeV]', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} p_T}$ [GeV$^{-1}$]', custom_bins=np.arange(0,110,5),ylogbool=True)
    histogram_multi_xsec([output['yg'], output2['yg']], [1.0, 1.0], 'yg', [r'HERWIG 7', r'Pyresias'], xlabel=r'Rapidity of emitted gluons', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} y}$', custom_bins=np.linspace(-3,3,50))
    histogram_multi_xsec([output['Eq'], output2['Eq']], [1.0, 1.0], 'Eq', [r'HERWIG 7', r'Pyresias'], xlabel=r'Energy of outgoing quarks', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} E}$', custom_bins=np.arange(0,120,2))
    histogram_multi_xsec([output['pzq'], output2['pzq']], [1.0, 1.0], 'pzq', [r'HERWIG 7', r'Pyresias'], xlabel=r'$p_z$ of outgoing quarks', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} p_z}$', custom_bins=np.arange(0,120,5))
    histogram_multi_xsec([output['pzg'], output2['pzg']], [1.0, 1.0], 'pzg', [r'HERWIG 7', r'Pyresias'], xlabel=r'$p_z$ of emitted gluons', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} p_z}$', custom_bins=np.arange(0,120,2))
    histogram_multi_xsec([output['Eg'], output2['Eg']], [1.0, 1.0], 'Eg', [r'HERWIG 7', r'Pyresias'], xlabel=r'Energy of emitted gluons', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} E}$', custom_bins=np.arange(0,220,2))
    histogram_multi_xsec([output['ng'], output2['ng']], [1.0, 1.0], 'ng', [r'HERWIG 7', r'Pyresias'], xlabel=r'number of emitted gluons', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} n_g}$ ', custom_bins=np.arange(0,15, 1))
    histogram_multi_xsec([output['minvg'], output2['minvg']], [1.0, 1.0], 'minvg', [r'HERWIG 7', r'Pyresias'], xlabel=r'invariant mass SQUARED of emitted gluons [GeV]', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} m_g}$ [GeV$^{-1}$]', custom_bins=np.arange(-20,20,1),ylogbool=True)



elif len(sys.argv) == 2:
    histogram_multi_xsec([output['pt']], [1.0], 'pt', ['LO'], xlabel=r'$p_T$ [GeV]', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} p_T}$ [GeV$^{-1}$]', custom_bins=np.arange(0,210, 10))
elif len(sys.argv) == 4:
    histogram_multi_xsec([output['pt'], output2['pt'], output3['pt']], [1.0, 1.0, 1.0], 'pt', ['LO', 'HW7', 'PYR'], xlabel=r'$p_T$ [GeV]', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} p_T}$ [GeV$^{-1}$]', custom_bins=np.arange(0,120, 5))
elif len(sys.argv) == 5:
    histogram_multi_xsec([output['pt'], output2['pt'], output3['pt'], output4['pt']], [1.0, 1.0, 1.0, 1.0], 'pt', ['LO', 'HW7', 'PYR', 'NLO'], xlabel=r'$p_T$ of quarks [GeV]', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} p_T}$ [GeV$^{-1}$]', custom_bins=np.arange(0,120, 5))
    histogram_multi_xsec([output['njets'], output2['njets'], output3['njets'], output4['njets']], [1.0, 1.0, 1.0, 1.0], 'njets', ['LO', 'HW7', 'PYR', 'NLO'], xlabel=r'$N_\mathrm{jets}$', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} N_\mathrm{jets}}$', custom_bins=np.arange(0,5,1))
    histogram_multi_xsec([output['ptj'], output2['ptj'], output3['ptj'], output4['ptj']], [1.0, 1.0, 1.0, 1.0], 'ptj', ['LO', 'HW7', 'PYR', 'NLO'], xlabel=r'$p_T$ of all jets [GeV]', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} p_T}$ [GeV$^{-1}$]', custom_bins=np.arange(0,200,5))
    histogram_multi_xsec([output['ptj1'], output2['ptj1'], output3['ptj1'], output4['ptj1']], [1.0, 1.0, 1.0, 1.0], 'ptj1', ['LO', 'HW7', 'PYR', 'NLO'], xlabel=r'$p_T$ of leading jet [GeV]', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} p_T}$ [GeV$^{-1}$]', custom_bins=np.arange(0,200,5))
    histogram_multi_xsec([output['ptj2'], output2['ptj2'], output3['ptj2'], output4['ptj2']], [1.0, 1.0, 1.0, 1.0], 'ptj2', ['LO', 'HW7', 'PYR', 'NLO'], xlabel=r'$p_T$ of sub-leading jet [GeV]', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} p_T}$ [GeV$^{-1}$]', custom_bins=np.arange(0,200,5))
    histogram_multi_xsec([output['sumpvecmag'], output2['sumpvecmag'], output3['sumpvecmag'], output4['sumpvecmag']], [1.0, 1.0, 1.0, 1.0], 'sumpvecmag', ['LO', 'HW7', 'PYR', 'NLO'], xlabel=r'$|\sum \vec{p}|$ [GeV]', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} |\sum \vec{p}|}$ [GeV$^{-1}$]', custom_bins=np.linspace(0,1,10000))
    histogram_multi_xsec([output['thrust'], output2['thrust'], output3['thrust'], output4['thrust']], [1.0, 1.0, 1.0, 1.0], 'thrust', ['LO', 'HW7', 'PYR', 'NLO'], xlabel=r'$T$', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} T}$', custom_bins=np.linspace(0,1,50))
    histogram_multi_xsec([output['ptg'], output2['ptg'], output3['ptg'], output4['ptg']], [1.0, 1.0, 1.0, 1.0], 'ptg', ['LO', 'HW7', 'PYR', 'NLO'], xlabel=r'$p_T$ of emitted gluons [GeV]', title=r'$e^+ e^- \rightarrow q\bar{q}$', ylabel=r'$\frac{1}{\sigma} \frac{\mathrm{d} \sigma}{\mathrm{d} p_T}$ [GeV$^{-1}$]', custom_bins=np.arange(0,120, 5))
