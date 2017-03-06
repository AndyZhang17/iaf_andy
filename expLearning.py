import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import numpy as np
import data as D
import utils.plotZ as plotZ

DATA_PATH = '/tmp/gmm1.npz'

data = D.loadData(DATA_PATH)
data_x = data['x']

## data visualisation
print('...data visualising')
plotZ.scatter2d(data_x[:,0],data_x[:,1])

import theano
import theano.tensor as T
import nodes
import nodes.iaf2 as iaf
import utils
import utils.mathT as mathT
import utils.mathZ as mathZ

f32 = theano.config.floatX
[data_N,DIM] = np.shape(data_x)
DECAY = 1e-3
LR0 = 0.1
MAXITER = 2000



## ----- ##
#  MODEL  #
## ----- ##
print('... constructing models')
lr = T.fscalar()
iafmodel = iaf.IafStack( 'IAF_MAP', dim=DIM )
layer1 = iaf.IafLinear('iaf1', DIM, lr=lr)
layerPer1 = iaf.IafPermute('per1', DIM)
layer2 = iaf.IafLinear('iaf2', DIM, lr=lr)
layerPer2 = iaf.IafPermute('per2', DIM)
layer3 = iaf.IafLinear('iaf3', DIM, lr=lr)
layerPer3 = iaf.IafPermute('per3', DIM)
layer4 = iaf.IafLinear('iaf4', DIM, lr=lr)

iafmodel.add([layer1, layerPer1])
iafmodel.add([layer2, layerPer2])
iafmodel.add([layer3, layerPer3])
iafmodel.add(layer4)

# symbolic initialisation
print('... symbolic initialisation')
x = utils.sharedf(data_x)
e = iafmodel.getNoise(data_N)
z,interz = iafmodel.forward(e, interout=True)

liksigmas = utils.sharedf(np.ones(DIM))
logp = mathT.gaussJoint(x,z,liksigmas,mean=True)
logq = iafmodel.mcLogQZ(e)
negELBO = logq - logp

iafmodel.setCost(negELBO)
updates = iafmodel.getUpdates()

print('... constructing functions')
trainq = theano.function([lr],negELBO,updates = updates)
traintheta = theano.function([lr],negELBO, updates=[(liksigmas,nodes.sgdParam(liksigmas,negELBO,lr))])
fz = theano.function([],z)
print('> Function Construction done')


i=0
while i<MAXITER:
    i+=1
    LR = LR0/(1+DECAY*i)
    cost = trainq(LR)
    cost2 = traintheta(LR)
    print(cost)

plotZ.hist2D(fz())
print(liksigmas.get_value())






