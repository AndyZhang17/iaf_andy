import matplotlib.pyplot as plt
import numpy as np


def scatter2d(x,y=None,type='b+',lims=None):
    if y is None:
        x,y = x[:,0],x[:,1]
    plt.plot(x,y,type)
    plt.show()


def hist_1d(x,bins=50,show=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    num, bins_, patches = ax.hist(x, bins, normed=1)
    plt.show()


def hist_2d(x,y,bins=50,show=False):
    heatmap, xedges, yedges = np.histogram2d(x,y,bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.figure()
    plt.clf()
    plt.imshow(heatmap.T,extent=extent,origin='lower')
    plt.show()


def line_2d(y,x=None,linetype='-',ylims=None,log=None,grid=False,title=None,xname='',yname='',legend=None):
    y = np.asarray(y)
    if y.ndim==1:
        y = np.reshape(y,(len(y),1))

    if x is None:
        x = range(y.shape[0])
    x = np.asarray(x)

    if x.ndim==1:
        x = np.reshape(x,(len(x),1))
    plt.figure()

    for d in range(np.shape(y)[1]):
        id_x = min(d,x.shape[1]-1)
        if legend is None:
            plt.plot( x[:,id_x], y[:,d], linetype )
        else:
            plt.plot( x[:,id_x], y[:,d], linetype, label=legend[d])
    if legend is not None:
        plt.legend()
    if title:
        plt.title(title)
    plt.grid(grid)
    plt.xlabel(xname)
    plt.ylabel(yname)
    if ylims!=None:
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,ylims[0],max(ylims[1],y2)))
    if log=='y':
        plt.yscale('log')
    elif log=='x':
        plt.xscale('log')
    else:
        pass

    plt.show()

def hist2D(p,bins=50,title=None):
    heatmap, xedges, yedges = np.histogram2d(p[:,0],p[:,1],bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.figure()
    plt.clf()
    plt.imshow(heatmap.T,extent=extent,origin='lower')
    plt.title(title)
    plt.show()