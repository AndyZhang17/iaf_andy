import numpy as np
import numpy.random as npr
import numpy.linalg as linalg
import utils
PI = utils.PI
INF = utils.INF

def upTranMask(side):
    mask = np.triu( np.ones((side,side)) )
    return mask


def lowTranMask(side):
    mask = np.tril( np.ones((side,side)) )
    return mask


def permutMat(side, enforcing=True):
    mat = npr.permutation(np.eye(side))
    while enforcing and (np.sum(np.abs(mat-np.eye(side)))==0):
        mat = npr.permutation(np.eye(side))
    return mat


def gaussInit(muin,varin):
    # single Gaussian
    try:
        d = len(muin)
        vardet, varinv = linalg.det(varin), linalg.inv(varin)
        logconst = -d/2.*np.log(2*PI) -0.5*np.log(vardet)
        def logP(x):
            submu = x-muin
            return logconst - 0.5*np.sum( submu*(np.dot(submu,varinv.T)), 1)
        def generator(size,dtype='float64'):
            return np.asarray( npr.multivariate_normal(muin,varin,size),dtype=dtype )
    except:
        def logP(x):
            logconst = -0.5 *np.log(2*PI) -0.5*np.log(varin)
            return logconst -0.5/varin*(x-muin)**2
        def generator(size,dtype='float64'):
            return np.asarray( npr.normal(muin,varin**.5,size),dtype=dtype )
    return logP, generator


def gaussMixInit(musin, varsin, probs):
    musin = np.asarray(musin)
    varsin = np.asarray(varsin)
    probs = np.asarray(probs)
    gs = list()
    if musin.ndim==1:
        num_g,d = np.shape(musin)[0],1
    else:
        num_g,d = np.shape(musin)
    gs = [ ( gaussInit(musin[i],varsin[i]) ) for i in range(num_g) ]
    def logP(x):
        logXs = np.array([g[0](x) for g in gs]).T
        probXs = np.exp(logXs)
        return np.log( np.sum( probXs*probs, 1) )
    def generator(size):
        indices = np.argmax( npr.multinomial(1,probs,size), axis=1 )
        return np.array( [ gs[id][1](1)[0] for id in indices ] )
    return logP, generator

