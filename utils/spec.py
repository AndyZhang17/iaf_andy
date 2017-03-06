import numpy as np
import numpy.random as npr

def gmmParam(numG,dim):
    mu = (npr.rand(numG,dim)*2-1)
    var = list()
    for n in range(numG):
        # R = npr.rand(3,dim)*2-1
        # var.append(np.dot(R.T,R))
        var.append(np.eye(dim)/5.)
    var = np.asarray(var)
    prob = npr.rand(numG)
    prob = prob/np.sum(prob)
    return mu,var,prob

