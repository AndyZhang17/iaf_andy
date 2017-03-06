import numpy as np
import numpy.random as npr
import theano
PI=np.pi
INF = 1e+8
ZERO = 1e-7
floatX = theano.config.floatX

def checkDictKeys(dictin,members,rerr=False):
    valid = np.prod(np.array([mem in dictin.keys() for mem in members])+0)
    if rerr and (not valid):
        raise ValueError('[ERR] Setup dict doesn\'t satisfy:\ndict: %s\nrequired: %s' % (repr(dictin),repr(members)))
    return valid

def sharedf(x, target=None, name=None,borrow=False):
    if target is None:
        return theano.shared(np.asarray(x,dtype=floatX), name=name, borrow=borrow, )
    else:
        return theano.shared(np.asarray(x,dtype=floatX), target=target, name=name, borrow=borrow)

import mathT as mathT
import mathZ as mathZ



def checkFile(filepath):
    import os.path as op
    return op.exists(filepath)

def checkDir(dir,build=True):
    import os
    flag = os.path.isdir(dir)
    if build and (not flag):
        os.makedirs(dir)
    return flag