import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import theano.tensor.nlinalg as Tlin
import numpy as np
import weights
import nodes as N
floatX = theano.config.floatX
import layers
import utils.mathz as M
ZERO = 1e-7

class IafLayer(layers.SimpleLayer):
    def __init__(self, setup):
        N.checkDictKeys(setup,['name'],rerr=True)
        super(IafLayer,self).__init__(setup)
    def forward(self,x):
        raise NotImplementedError('base class being called. %s' % (self.name))
    def logDetJacobian(self):
        raise NotImplementedError('base class being called. %s' % (self.name))

class IafStack(layers.LayerStack):
    def __init__(self,name):
        super(IafStack,self).__init__(name)
    def logDetJacoSum(self):
        totalJaco = N.sharedScalar(0)
        for layer in self.layers:
            totalJaco = totalJaco + layer.logDetJacobian()
        return totalJaco

class IafSigmoid(IafLayer):
    def __init__(self,setup):
        super(IafSigmoid, self).__init__(setup)
        self.active = T.fmatrix()
    def forward(self,x):
        self.forwardflag = True
        self.active = nnet.sigmoid(x)
        return self.active
    def logDetJacobian(self):
        #if not self.forwardflag:
        #    raise Exception('sigmoid %s: logDetJacobian() called before forward()' %(self.name))
        return T.mean(T.log(ZERO+T.abs_(self.active*(1.-self.active))))


class IafPermute(IafLayer):
    def __init__(self,setup):
        N.checkDictKeys(setup,['dim'],rerr=True)
        super(IafPermute,self).__init__(setup)
        self.initWeight()
    def intiWeight(self):
        self.w = M.permutMat(self.dim)
        self.detJ = np.abs(np.linalg.det(self.w))
    def forward(self,x):
        return T.dot(x,self.w)
    def logDetJacobian(self):
        return np.log(self.detJ)


class IafLinear(IafLayer):
    def __init__(self, setup):
        N.checkDictKeys(setup,['dim','bias','initmethod'],rerr=True)
        super(IafLinear,self).__init__(setup)
        self.weightsmask = M.upTranMask(self.dim)
        self.initWeight()
    def initWeight(self):
        if self.initmethod =='uniform':
            self.w = weights.linearInitUnif(self.dim,self.dim)
        elif self.initmethod=='identity':
            self.w = weights.iafLinearInitIdentity(self.dim)
        elif self.initmethod=='test1':
            self.w = weights.iafLinearTest1(self.dim)
        elif self.initmethod=='test2':
            self.w = weights.iafLinearTest2(self.dim)
        else:
            raise Exception('Unknown initialization method. %s' % (self.initmethod))
        if self.bias:
            self.b = weights.biasInit(self.dim)
    def forward(self,x):
        y = T.dot(x,self.weightsmask*self.w)
        if self.bias:
            y = y+self.b
        return y
    def logDetJacobian(self):
        diags = Tlin.extract_diag(self.w)
        return T.sum( T.log( T.abs_(diags) ) )





