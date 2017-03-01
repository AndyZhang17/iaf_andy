import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import theano.tensor.nlinalg as Tlin

import weights
import nodes as N
import layers
import utils.mathZ as mathZ
import utils
import utils.mathT as mathT
import numpy as np
import numpy.random as npr

floatX = theano.config.floatX
ZERO = utils.ZERO

class IafLayer(layers.SimpleLayer):
    def __init__(self, setup):
        super(IafLayer,self).__init__(setup)

    def logDetJacobian(self):
        '''return sym of mean(log|Jaco|)'''
        raise NotImplementedError('base class being called. %s' % (self.name))


class IafStack(layers.LayerStack):
    def __init__(self,name,dim):
        super(IafStack,self).__init__(name)
        self.dimin = self.dimout= self.dim = dim
        self.initLogPrior()

    def initLogPrior(self):
        priorvar = np.eye(self.dim)
        self.logPrior = mathT.gaussInit(np.zeros(self.dim), priorvar)
        _, self.priorGen = mathZ.gaussInit(np.zeros(self.dim),priorvar)

    def logDetJacoSum(self):
        totaljaco = N.sharedScalar(0)
        for layer in self.layers:
            totaljaco = totaljaco + layer.logDetJacobian()
        return totaljaco

    def costMCLogQZ(self,esamples):
        logprior = T.mean(self.logPrior(esamples),axis=0)
        return logprior - self.logDetJacoSum()

    def showParam(self):
        out = list()
        for layer in self.layers:
            out.append(layer.showParam())
        return out


class IafSigmoid(IafLayer):
    def __init__(self,setup):
        super(IafSigmoid, self).__init__(setup)
        self.active = T.fmatrix()
        self.fwdflag = False

    def forward(self,x):
        self.fwdflag = True
        self.active = nnet.sigmoid(x)
        return self.active

    def logDetJacobian(self):
        if not self.fwdflag:
            raise Exception('sigmoid %s: logDetJacobian() called before forward()' %(self.name))
        grads = self.active*(1.-self.active)
        return T.mean( T.sum( T.log( T.abs_(grads)+ZERO ), axis=1 ) )


class IafTanh(IafLayer):
    def __init__(self,setup):
        super(IafLayer, self).__init__(setup)
        self.x = T.fmatrix()
        self.active = T.fmatrix()

    def forward(self,x):
        self.x = x
        self.active = T.tanh(self.x)
        return self.active

    def logDetJacobian(self):
        grads = 1./( T.sqr(T.cosh(self.x))+ZERO )
        return T.mean( T.sum( T.log( T.abs_(grads)+ZERO) ,axis=1 ) )



class IafPermute(IafLayer):
    def __init__(self,setup):
        N.checkDictKeys(setup,['dim'],rerr=True)
        super(IafPermute,self).__init__(setup)
        self.dimin = self.dimout = self.dim
        self.initWeight()

    def initWeight(self):
        self.w = N.sharedf(mathZ.permutMat(self.dim))
        self.detJ = N.sharedsfGpu(0.)

    def setWeight(self,weight):
        self.w = weight

    def forward(self,x):
        return T.dot(x,self.w)

    def logDetJacobian(self):
        return self.detJ


class IafNormalising(IafLayer):
    def __init__(self,setup):
        N.checkDictKeys(setup,['dim','lr'],rerr=True)
        super(IafLayer, self).__init__(setup)
        self.dimin = self.dimout = self.dim
        self.b = N.sharedScalar(0.)
        self.x = T.fmatrix()
        # self.w = N.sharedf(npr.rand(self.dim))
        self.u = N.sharedf(npr.rand(self.dim))
        self.w = N.sharedf([0.]*self.dim)
        # self.u = N.sharedf( ((npr.rand(self.dim)>0.5)-0.5)/50. )

    def showParam(self):
        return [self.w.eval(),self.b.eval(),self.u.eval()]

    def forward(self,x):
        self.x = x
        self.active = T.tanh( T.dot(self.x,self.w)+self.b )
        return T.outer( self.active, self.u ) + x
    def logDetJacobian(self):
        dtrans = 1./(T.cosh(self.active)**2)
        # dtrans = T.grad(T.sum(self.active),wrt=self.x)
        return T.mean(T.log(T.abs_(1.+T.dot(self.u,self.w)*dtrans )))
    def getUpdates(self,cost):
        def newP(cost,param):
            return param-self.lr*T.grad(cost,wrt=param)
        return [(p,newP(cost,p)) for p in [self.w,self.b,self.u]]


class IafLinear(IafLayer):
    def __init__(self, setup):
        N.checkDictKeys(setup,['dim','bias','initmethod','lr'],rerr=True)
        super(IafLinear,self).__init__(setup)
        self.dimin = self.dimout = self.dim
        self.weightsmask = mathZ.upTranMask(self.dim)
        self.initWeight()

    def initWeight(self):
        if self.initmethod =='uniform':
            self.w = weights.linearInitUnif(self.dimin,self.dimout)
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

    def setWeight(self,weights,bias=None):
        self.w = N.sharedf(np.asarray(weights))
        if bias is not None:
            self.bias = True
            self.b = N.sharedf(np.asarray(bias))

    def forward(self,x):
        y = T.dot(x,self.weightsmask*self.w)
        if self.bias:
            y = y+self.b
        return y

    def logDetJacobian(self):
        diags = Tlin.extract_diag(self.w)
        return T.sum( T.log( T.abs_(diags) ) )

    def getUpdates(self, cost):
        def newP(cost,param):
            return param - self.lr*T.grad(cost,wrt=param)
        if self.bias:
            return [(p,newP(cost,p)) for p in [self.w, self.b]]
        else:
            return [(self.w,newP(cost,self.w))]

    def getGrads(self,cost):
        if self.bias:
            return [T.grad(cost,wrt=p) for p in [self.w,self.b]]
        else:
            return [T.grad(cost,wrt=self.w)]

    def showParam(self):
        if self.bias:
            return [self.w.eval(),self.b.eval()]
        else:
            return [self.w.eval()]







