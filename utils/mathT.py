import theano
import theano.tensor as T
import numpy as np
import numpy.linalg as linalg
import theano.tensor.nlinalg as nlinalg
import utils

PI = utils.PI
ZERO = utils.ZERO
floatX = utils.floatX

def gaussJoint(x,y,sigmas,numExp=1,mean=False):
    '''input all symbolic'''
    sigma2s = T.sqr(sigmas)
    log_const = -x.shape[1]/2.*np.log(2*PI) -.5*T.sum(T.log(sigma2s))
    diff = y-T.repeat(x,numExp,axis=0)
    log_subs = -.5* T.sum(T.sqr(diff)/sigma2s,axis=1)
    if mean:
        return T.mean(log_const+log_subs)
    return log_const+log_subs



def gaussInit(muin,varin,mean=False):
    d = muin.shape[0]
    vardet, varinv = nlinalg.det(varin), nlinalg.matrix_inverse(varin)
    logconst = -d/2.*np.log(2*PI) - .5*T.log(vardet)
    def logP(x):
        submu = x-muin
        out = logconst - .5*T.sum(submu*(T.dot(submu, varinv.T)),axis=1)
        if mean:
            return T.mean(out)
        return out
    return logP


# multivariate Gaussian
def gaussInit0(muin,varin,mean=False):
    muin, varin = np.asarray(muin), np.asarray(varin)
    d = len(muin)
    vardet, varinv = linalg.det(varin), linalg.inv(varin)
    logconst = -d/2.*np.log(2*PI) -.5*np.log(vardet)
    def logP(x):
        submu = x-muin
        out = logconst -.5*T.sum(submu*(T.dot(submu,varinv.T)),axis=1)
        if mean:
            return T.mean(out)
        return out
    return logP


def gaussMixInit(musin, varsin, probs, mean=False):
    musin,varsin,probs  = np.asarray(musin),np.asarray(varsin),np.asarray(probs)
    numgauss,dim = np.shape(musin)
    gs = [ gaussInit(musin[i],varsin[i],mean=False)  for i in range(numgauss) ]
    def logP(x):
        indprobs = T.cast([ T.exp(gs[i](x))*probs[i] for i in range(numgauss) ],dtype=floatX)
        xprobs = T.sum(indprobs,axis=0)
        out = T.log(xprobs)
        if mean:
            return T.mean(out)
        return out
    return logP

def compAbs(x, offset):
    return T.abs_(x)>=offset

def coshApx(x, offset=1.5):
    out = T.switch(compAbs(x,offset), .5*T.exp(T.abs_(x)), T.cosh(x) )
    return out

def coshsqrApx(x, offset=1.5):
    out = T.switch(compAbs(x,offset), .25*T.exp(2*T.abs_(x)), T.sqr(T.cosh(x)))
    return out

def tanhApx(x, offset=1.5):
    small = T.tanh(x)
    sign = T.sgn(x)
    large = sign*(1-2*T.exp(-2*sign*x))
    return T.switch( compAbs(x,offset), large, small )
