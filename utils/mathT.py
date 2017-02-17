__author__ = 'andy17'


import theano
import theano.tensor as T
import numpy as np
import numpy.linalg as linalg
PI = np.pi


# multivariate Gaussian
def gaussInit(muin,varin):
    d = len(muin)
    vardet, varinv = linalg.det(varin), linalg.inv(varin)
    logconst = -d/2.*np.log(2*PI) -.5*np.log(vardet)
    def logP(x):
        submu = x-muin
        return logconst -.5*T.sum(submu*(T.dot(submu,varinv.T)),1)
    return logP

def gaussMixInit(musin, varsin,probs):
    pass
