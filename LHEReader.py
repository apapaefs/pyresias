import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
import math
from tqdm import tqdm
import matplotlib.gridspec as gridspec # more plotting 
import gzip

##########################
# FUNCTIONS
##########################

# Functions to handle colors in matplotlib plots:
# choose the next colour -- for plotting
ccount = 0
def next_color():
    global ccount
    colors = ['green', 'orange', 'red', 'blue', 'black', 'cyan', 'magenta', 'brown', 'violet'] # 9 colours
    color_chosen = colors[ccount]
    if ccount < 8:
        ccount = ccount + 1
    else:
        ccount = 0    
    return color_chosen
# do not increment colour in this case:
def same_color():
    global ccount
    colors = ['green', 'orange', 'red', 'blue', 'black', 'cyan', 'magenta', 'brown', 'violet'] # 9 colours
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
def histogram_multi_xsec_stacked(DATA_array, CrossSection_array, plot_type, plotnames_multi, xlabel='', ylabel='fraction/bin', nbins=50, title='', custom_bins=[], ylogbool=False, xlogbool=False, outputfiletag=''):
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
            Y = np.multiply(np.array([bins,bins]).T.flatten(),CrossSection_array[dd])
            binstot = bins/float(len(DATA))*CrossSection_array[dd]
        else: 
            Y = Y + np.multiply(np.array([bins,bins]).T.flatten(),CrossSection_array[dd])
            binstot = binstot + bins/float(len(DATA))*CrossSection_array[dd]
        dd = dd+1
    center = (edges[:-1] + edges[1:]) / 2
    plt.plot(X,Y, label='total', color=next_color(), lw=1)
    #plt.errorbar(X, Y, yerr=., color=same_color(), lw=0, elinewidth=1, capsize=1)


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
        infile = plot_type + '-' + outputfiletag + '_stacked.dat'
    else:
        infile = plot_type + '_stacked.dat'
    print('output in', infile.replace('.dat','.pdf'))
    plt.savefig(infile.replace('.dat','.pdf'), bbox_inches='tight')
    #plt.close(fig)
    plt.show()


# function to plot histograms: including the cross section normalization
# CrossSection_array contains ARRAYS of cross sections 
# DATA_array contains ARRAYS of data for each event. Each array represents a different type of input (e.g. a run with different parameters, etc.). 
# plot_type is simply the main name of the plot
# plotnames_multi has to be an array of equal size to DATA_array
# custom_bins can be provided for the desired observable
def histogram_multi_xsec(DATA_array, CrossSection_array, plot_type, plotnames_multi, xlabel='', ylabel='fraction/bin', nbins=50, title='', custom_bins=[], ylogbool=False, xlogbool=False, outputfiletag=''):
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
        Y = np.multiply(np.array([bins,bins]).T.flatten(),CrossSection_array[dd])
        
        plt.plot(X,Y, label=plotnames_multi[dd], color=next_color(), lw=1)
        #center = (edges[:-1] + edges[1:]) / 2
        #plt.errorbar(center, bins, yerr=errors, color=same_color(), lw=0, elinewidth=1, capsize=1)
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
    #plt.close(fig)
    plt.show()

    
# function to plot histograms
# DATA_array contains ARRAYS of data for each event. Each array represents a different type of input (e.g. a run with different parameters, etc.). 
# plot_type is simply the main name of the plot
# plotnames_multi has to be an array of equal size to DATA_array
# custom_bins can be provided for the desired observable
def histogram_multi(DATA_array, plot_type, plotnames_multi, xlabel='', ylabel='fraction/bin', nbins=50, title='', custom_bins=[], ylogbool=False, xlogbool=False, outputfiletag=''):
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
    plt.show()

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
