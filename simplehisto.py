from matplotlib import pyplot as plt # plotting
import matplotlib.gridspec as gridspec # more plotting
from matplotlib.ticker import MultipleLocator
import numpy as np
from matplotlib import ticker 


def simplehisto(message, plot_type, outputdirectory, array_to_hist, xlab, ylab, nbins=50, xlog=False, ylog=False, norm=True):

    ############
    print('---')
    print(message)
    # plot settings ########
    # plot:
    # plot settings
    ylog = False
    xlog = False

    # construct the axes for the plot
    gs = gridspec.GridSpec(4, 4)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(False)

    # get the histogram bins:
    bins, edges = np.histogram(array_to_hist, bins=nbins)
    left,right = edges[:-1],edges[1:]
    X = np.array([left,right]).T.flatten()
    Y = np.array([bins,bins]).T.flatten()

    # normalise:
    if norm:
        Y = Y/np.linalg.norm(Y)
    # plot
    plt.plot(X,Y, label='', color='red', lw=1)

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

    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # save the figure
    print('saving the figure')
    # save the figure in PDF format
    infile = plot_type + '.dat'
    print('---')
    print('output in', outputdirectory + infile.replace('.dat','.pdf'))
    plt.savefig(outputdirectory + infile.replace('.dat','.pdf'), bbox_inches='tight')
    plt.close(fig)
