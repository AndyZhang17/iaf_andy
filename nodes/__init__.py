import theano
import theano.tensor as T
import numpy as np
import numpy.random as npr
import sys

sys.setrecursionlimit(10000)
floatX = theano.config.floatX

#rng = theano.tensor.shared_randomstreams.RandomStreams(0)

#def cast(x):


def sharedScalar(x,name=None):
    return theano.shared(x,name=name)

def sharedsfGpu(x,name=None):
    return T.cast( theano.shared(x,name=name), dtype=floatX )

def sharedf(x, target=None, name=None,borrow=False):
    if target is None:
        return theano.shared(np.asarray(x,dtype=floatX), name=name, borrow=borrow, )
    else:
        return theano.shared(np.asarray(x,dtype=floatX), target=target, name=name, borrow=borrow)

def sharedfRand(stddev,size):
    return sharedf(npr.normal(0,stddev,size=size))

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
    def __repr__(self):
        return '<%s>'%str('\n'.join('%s: %s'% (k,repr(v)) for (k,v) in self.__dict__.iteritems()))

def checkDictKeys(dictin,members,rerr=False):
    valid = np.prod(np.array([mem in dictin.keys() for mem in members])+0)
    if rerr and (not valid):
        raise ValueError('Input dict doesn\'t satisfy:\ndict: %s\nrequired: %s' % (repr(dictin),repr(members)) )
    return valid


def sgdParam(param,cost,lr):
    return param - lr*T.grad(cost,wrt=param)



import layers as layers
import iaf as iaf
import weights as weights

