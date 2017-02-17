import theano
import theano.tensor as T
import numpy as np
import weights
import theano.tensor.nnet as nnet
import nodes as N
import utils.mathz as utilsM

class SimpleLayer(object):
    def __init__(self, setup):
        # setup: dictionary of parameters
        N.checkDictKeys(setup,['name'],rerr=True)
        self.__dict__.update(setup)
    def forward(self,x):
        raise NotImplementedError('base class being called. %s' % (self.name))


class LayerStack(object):
    def __init__(self,name):
        self.name = name
        self.layers = list()
    def forward(self,x):
        y = self.layers[0].forward(x)
        for i in range(1,len(self.layers)):
            y = self.layers[i].forward(y)
        return y
    def add(self,newlayer):
        self.layers.append(newlayer)
    def __repr__(self):
        lnames = '\n'.join( [ str(i)+' < '+self.layers[i].name+' >' for i in range(len(self.layers)) ]  )
        return '\n'.join([self.name, lnames])


class LinearLayer(SimpleLayer):
    def __init__(self, setup):
        # setup: name, dimIn, dimOut, batchSize, bias=True, initMethod='uniform'):
        N.checkDictKeys(setup,['name', 'dimin', 'dimout', 'batchsize', 'bias', 'initmethod'],rerr=True)
        super(LinearLayer, self).__init__(setup)
        self.initWeight()
    def initWeight(self):
        if self.initmethod == 'uniform':
            self.w = weights.linearInitUnif(self.dimin,self.dimout)
        else:
            raise Exception('Unknown initialization method. %s' % (self.initmethod))
        if self.bias:
            self.b = weights.biasInit(self.dimout)
    def forward(self,x):
        y = T.dot(x,self.w)
        if self.bias:
            y = y+self.b
        return y



class SigmoidLayer(SimpleLayer):
    def __init__(self,setup):
        super(SigmoidLayer,self).__init__(setup)
    def forward(self,x):
        return nnet.sigmoid(x)

class PermuteLayer(SimpleLayer):
    def __init__(self,setup):
        N.checkDictKeys(setup,['dim'],rerr=True)
        super(PermuteLayer,self).__init__(setup)
        self.initWeight()
        self.detjacobian = np.abs(np.linalg.det(self.w))
    def initWeight(self):
        self.w = utilsM.permutMat(self.dim)
    def forward(self,x):
        return T.dot(x,self.w)
    def backward(self,x):
        return T.dot(x,self.w.T)
    def detJacobian(self):
        return self.detjacobian


