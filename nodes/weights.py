import numpy as np
import numpy.random as npr
import nodes as Nd
import utils.mathZ as M


def biasInitZero(dimout, name=None):
    return Nd.sharedf( np.zeros(dimout), name=name )

def biasInit(dimout, name=None):
    return Nd.sharedf( np.zeros(dimout), name=name )

def biasInitRandn(dimout, mean, scale, name=None):
    bias = npr.randn(dimout)*float(scale) + float(mean)
    return Nd.sharedf( bias, name=name )


def linearInitUnif(dimin,dimout,name=None):
    low = -np.sqrt(6./(dimin+dimout))
    high = np.sqrt(6./(dimin+dimout))
    weights= npr.rand(dimin,dimout)*(high-low)-low
    return Nd.sharedf( weights, name=name )

def linearInitGauss(dimin, dimout,name=None):
    weights = npr.randn(dimin, dimout)*np.sqrt(6./(dimin+dimout))
    return Nd.sharedf( weights, name=name )

def iafLinearInitIdentity(dim,name=None):
    weights = np.eye(dim)
    return Nd.sharedf( weights, name=name )

def iafLinearTest1(dim,name=None):
    weights = M.upTranMask(dim)
    return Nd.sharedf(weights,name=name)

def iafLinearTest2(dim,name=None):
    weights = M.upTranMask(dim)*2.
    return Nd.sharedf(weights,name=name)


def autoregMaskL(dim,name=None):
    mask = M.upTranMask(dim)
    return Nd.sharedf(mask, name=name)

def diagMaskL(dim,name=None):
    mask = np.eye(dim)
    return Nd.sharedf(mask,name=name)

def linAutoregInitGauss(dim, scale=0.1, name=None):
    mask = M.upTranMask(dim)
    weights = npr.randn(dim, dim)*np.sqrt(scale/(dim+dim))*mask
    return Nd.sharedf( weights, name=name )

def linScaleInit(dim,noise=True,name=None):
    scale = np.array([1.]*dim)
    if noise:
        scale  = scale + (npr.rand(dim)*.1-0.05)
    return Nd.sharedf( scale, name=name )