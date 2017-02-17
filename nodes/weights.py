import numpy as np
import numpy.random as npr
import nodes as Nd
import utils.mathz as M


def biasInit(dimout, name=None):
    return Nd.sharedf( np.zeros(dimout), name=name )

def linearInitUnif(dimin,dimout,name=None):
    low = -np.sqrt(6./(dimin+dimout))
    high = np.sqrt(6./(dimin+dimout))
    weights= npr.rand(dimin,dimout)*(high-low)-low
    return Nd.sharedf( weights, name=name )

def iafLinearInitIdentity(dim,name=None):
    weights = M.upTranMask(dim)
    sums = np.sum(weights,axis=0)
    weights = weights/sums
    return Nd.sharedf( weights, name=name )

def iafLinearTest1(dim,name=None):
    weights = M.upTranMask(dim)
    return Nd.sharedf(weights,name=name)

def iafLinearTest2(dim,name=None):
    weights = M.upTranMask(dim)*2.
    return Nd.sharedf(weights,name=name)


