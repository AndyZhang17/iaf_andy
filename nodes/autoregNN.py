__author__ = 'andy17'
import nodes as N
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import weights
import utils.mathT as mathT

class AutoregNNLinear(object):
    def __init__(self,dim,lr):
        self.lr = lr
        self.dimin = self.dimout = dim
        self.mask = weights.autoregMask(self.dimin)
        self.w1 = weights.linAutoregInitGauss(self.dimin,name='wm') * self.mask
        self.w2 = weights.linAutoregInitGauss(self.dimin,name='ws') * self.mask
        self.b1 = weights.biasInitZero(self.dimout,name='bm')
        self.b2 = weights.biasInitZero(self.dimout,name='bs')
        self.meanLogSig = T.fscalar()
        self.cost = T.fscalar()
        self.params = [ self.w1, self.b1, self.w2, self.b2 ]
        self.uptflg = True

    def forward(self,x):
        m = T.dot(x, self.w1)+self.b1  # NxD
        sig =nnet.sigmoid( T.dot(x,self.w2) + self.b2 )  # NxD
        self.meanLogSig = T.mean( T.sum( T.log(sig), axis=1 ) )   # mean( N )
        return sig*x + (1.-sig)*m

    def logDetJaco(self):
        return self.meanLogSig

    def setCost(self,cost):
        self.cost=cost

    def getUpdates(self):
        return [ (p,N.newParam(p,self.cost,self.lr)) for p in self.params ]

    def getGrads(self):
        return [ T.grad(self.cost,wrt=p) for p in self.params ]

    def showParam(self):
        def evalParam(p):
            f = theano.function(inputs=[],outputs=p)
            return f()
        return [ evalParam(p) for p in self.params ]


class AutoregStack(object):
    def __init__(self,name):
        self.name=name
        self.layers = list()
        self.logPrior = mathT.gaussInit(np.zeros(self.dim), np.e)

    def forward(self,x):
        y = self.layers[0].forward(x)
        if len(self.layers)>1:
            for layer in self.layers[1:]:
                y = layer.forward(y)
        return y

    def add(self,layer):
        self.layers.append(layer)

    def setCost(self,cost):
        for layer in self.layers:
            if layer.uptflg:
                layer.setCost(cost)

    def getUpdates(self):
        updates = list()
        for layer in self.layers:
            if layer.uptflg:
                updates.extend(layer.getUpdates())
        return updates

    def getGrads(self):
        grads = list()
        for layer in self.layers:
            if layer.uptflg:
                grads.extend(layer.getGrads())
        return grads

    # def initLogPrior(self):





