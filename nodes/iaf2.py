import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import theano.tensor.nlinalg as Tlin
import weights
import nodes as N
import utils
import utils.mathZ as mathZ
import utils.mathT as mathT
import numpy as np
import numpy.random as npr
floatX = theano.config.floatX
ZERO = utils.ZERO

class IafStack(object):
    def __init__(self, name, dim):
        self.name = name
        self.dimin = self.dimout = dim
        self.layers = []
        self.initLogPrior()

    def initLogPrior(self):
        var = np.eye(self.dimin)*3.
        self.meanLogPrior = mathT.gaussInit(np.zeros(self.dimin), var, mean=True)
        _, self.eGen  = mathZ.gaussInit(np.zeros(self.dimin), var)

    def forward(self,x,interout=False):
        y = self.layers[0].forward(x)
        intery = [y]
        if len(self.layers)>1:
            for layer in self.layers[1:]:
                y = layer.forward(y)
                intery.append(y)
        if interout:
            return y, intery
        return y

    def mcLogQZ(self,esamples):
        totallogjaco = N.sharedScalar(0.)
        for layer in self.layers:
            totallogjaco = totallogjaco + layer.logDetJaco()
        return self.meanLogPrior(esamples) - totallogjaco

    def add(self,layer):
        self.layers.append(layer)

    def setCost(self,cost):
        for layer in self.layers:
            layer.setCost(cost)

    def getUpdates(self):
        updates = list()
        for layer in self.layers:
            updates.extend(layer.getUpdates())
        return updates

    def getParams(self):
        params = list()
        for layer in self.layers:
            params.extend(layer.params)
        return params

    def getGrads(self, cost=None):
        grads = list()
        for layer in self.layers:
            grads.extend(layer.getGrads(cost=cost))
        return grads

    def getValueParams(self):
        values = list()
        for layer in self.layers:
            values.extend(layer.getValueParams())
        return values



class IafLayer(object):
    def __init__(self,name):
        self.name=name
        self.params = []

    def setCost(self,cost):
        pass

    def forward(self,x):
        return x

    def getUpdates(self):
        return []

    def getGrads(self,cost=None):
        return []

    def logDetJaco(self):
        return 0.

    def getValueParams(self):
        return []


class IafPermute(IafLayer):
    def __init__(self,name,dim):
        super(IafPermute,self).__init__(name)
        self.dimin = self.dimout = dim
        self.w = N.sharedf( mathZ.permutMat(self.dimin,enforcing=True) )

    def setWeight(self,weight):
        self.w = N.sharedf( np.asarray(weight) )

    def forward(self,x):
        return T.dot(x,self.w)


class IafLinear(IafLayer):
    def __init__(self, name, dim, lr):
        super(IafLinear,self).__init__(name)
        self.lr = lr
        self.dimin = self.dimout = dim
        self.mask = weights.autoregMaskL(self.dimin)
        # self.mask = weights.diagMaskL(self.dimin)

        self.w = weights.linAutoregInitGauss(self.dimin, scale=0.1,name='w')
        # self.b = weights.biasInitZero(self.dimout,name='b')
        self.b = weights.biasInitRandn(self.dimout, mean=0, scale=.01, name='b')
        self.u = weights.biasInitRandn(self.dimout, mean=0, scale=.01, name='u')

        self.params = [self.w, self.b, self.u]
        self.meanlogdetjaco = T.fscalar()

    def setParams(self,w,b,u):
        w,b,u = np.asarray(w), np.asarray(b), np.asarray(u)
        self.w = N.sharedf(w,name='w')
        self.b = N.sharedf(b,name='b')
        self.u = N.sharedf(u,name='u')
        self.params = [self.w, self.b, self.u]

    def forward(self,x):
        a = T.dot(x,self.w*self.mask) + self.b   # NxD
        wdiag = Tlin.extract_diag( self.w )
        # cosha = mathT.coshApx(a)
        coshsqr = mathT.coshsqrApx(a)
        # coshsqr = T.sqr( T.cosh(a) )

        self.meanlogdetjaco = T.mean( T.sum( T.log( T.abs_( 1.+ self.u*wdiag/coshsqr ) ), axis=1 ) )
        return x + self.u * mathT.tanhApx( a )
        # return x + self.u * T.tanh( a )

    def logDetJaco(self):
        return self.meanlogdetjaco

    def setCost(self,cost):
        self.cost = cost

    def getUpdates(self):
        return [ (p,N.newParam(p,self.cost,self.lr)) for p in self.params]

    def getGrads(self,cost=None):
        if cost is not None:
            return [ T.grad(cost,wrt=p) for p in self.params ]
        return [ T.grad(self.cost,wrt=p) for p in self.params ]

    def getValueParams(self):
        def evalParam(p):
            f = theano.function(inputs=[],outputs=p)
            return f()
        return [evalParam(p) for p in self.params]




