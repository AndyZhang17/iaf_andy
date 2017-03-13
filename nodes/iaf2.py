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
    def __init__(self, name, dim, lr):
        self.name = name
        self.dimin = self.dimout = self.dim = dim
        self.lr = lr
        self.layers = []
        self.priorstd = N.sharedScalar( (4./self.dim)**.5 )
        self.initLogPrior()

    def initLogPrior(self):
        noisevar = N.sharedf(np.eye(self.dim))*T.sqr(self.priorstd)
        noisemu = N.sharedf(np.zeros(self.dim))
        self.logPrior = mathT.gaussInit(noisemu, noisevar)

    def getNoise(self,num):
        from theano.tensor.shared_randomstreams import RandomStreams as trands
        trng = trands()
        return trng.normal((num,self.dim)) * self.priorstd

    def forward(self,x,interout=False):
        intery = [self.layers[0].forward(x)]
        for layer in self.layers[1:]:
            intery.append( layer.forward(intery[-1]) )
        if interout:
            return intery[-1], intery
        return intery[-1]

    def mcLogQZ(self,esamples):
        totallogjaco = self.layers[0].logDetJaco()
        for layer in self.layers[1:]:
            totallogjaco = totallogjaco + layer.logDetJaco()
        self.meanlogprior = T.mean(self.logPrior(esamples))
        return self.meanlogprior - totallogjaco

    def add(self,layer):
        if isinstance(layer,list) or isinstance(layer,tuple):
            self.layers.extend(layer)
        else:
            self.layers.append(layer)

    def setCost(self,cost):
        self.cost = cost
        for layer in self.layers:
            layer.setCost(self.cost)

    def getParams(self):
        params = list()
        for layer in self.layers:
            params.extend(layer.params)
        return params

    def getUpdates(self):
        updates = list()
        for layer in self.layers:
            updates.extend(layer.getUpdates())
        return updates

    def getParamShapes(self):
        shapes = list()
        for layer in self.layers:
            shapes.extend(layer.paramshapes)
        return shapes

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
        self.paramshapes = []

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
        '''
        out = x + tanh( x*w + b )
        :param name: str
        :param dim:  int, dimension of the input nodes
        :param lr:   theano symbolic, learning rate
        :return:
        '''
        super(IafLinear,self).__init__(name)
        self.lr = lr
        self.dimin = self.dimout = dim
        self.mask = weights.autoregMaskL(self.dimin)

        scale = (.0002/self.dimin)**0.5
        self.w = weights.linAutoregInitGauss(self.dimin, scale=scale,name='w')
        self.b = weights.biasInitRandn(self.dimout, mean=0, scale=scale, name='b')
        self.u = weights.biasInitRandn(self.dimout, mean=0, scale=scale, name='u')

        self.params = [self.w, self.b, self.u]
        self.paramshapes = [(dim,dim),(dim,),(dim,)]
        self.wdiag = Tlin.extract_diag( self.w )
        self.meanlogdetjaco = T.fscalar()
        self.cost = T.fscalar()

    def setParams(self, w, b, u):
        '''
        DON'T USE
        should replace the values of w,b,u in-place
        '''
        w,b,u = np.asarray(w), np.asarray(b), np.asarray(u)
        self.w = N.sharedf(w, name='w')
        self.b = N.sharedf(b, name='b')
        self.u = N.sharedf(u, name='u')
        self.params = [self.w, self.b, self.u]

    def forward(self,x):
        '''
        passing x through this iaf layer
        :param x: symbolic
        :return:  symbolic
        '''
        a = T.dot(x, self.w*self.mask) + self.b   # NxD
        # coshsqr = mathT.coshsqrApx(a)
        coshsqrinv = mathT.coshsqrinvApx(a)
        self.meanlogdetjaco = T.mean( T.sum( T.log( T.abs_( 1.+ self.u*self.wdiag*coshsqrinv ) ), axis=1 ) )
        return x + self.u *mathT.tanhApx(a)

    def logDetJaco(self):
        return self.meanlogdetjaco

    def setCost(self,cost):
        self.cost = cost

    def getUpdates(self):
        return [ (p,N.sgdParam(p,self.cost,self.lr)) for p in self.params]

    def getGrads(self,cost=None):
        if cost is not None:
            return [ T.grad(cost,wrt=p) for p in self.params ]
        return [ T.grad(self.cost,wrt=p) for p in self.params ]

    def getValueParams(self):
        return [ p.get_value() for p in self.params]



def experiment(lr,dim):
    Qexp = IafStack('IAF_MAP', dim=dim, lr=lr)

    l1 = IafLinear('iaf1', dim, lr=lr)
    p1 = IafPermute('per1', dim)
    l2 = IafLinear('iaf2', dim, lr=lr)
    p2 = IafPermute('per2', dim)
    l3 = IafLinear('iaf3', dim, lr=lr)
    p3 = IafPermute('per3', dim)
    l4 = IafLinear('iaf4', dim, lr=lr)
    p4 = IafPermute('per4', dim)
    l5 = IafLinear('iaf5', dim, lr=lr)

    Qexp.add([l1,p1,l2,p2,l3,p3,l4,p4,l5])
    return Qexp
