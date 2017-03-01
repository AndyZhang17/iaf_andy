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

import mathT as mathT
import mathZ as mathZ