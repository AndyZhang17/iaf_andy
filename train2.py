__author__ = 'andy17'


import theano
import theano.tensor as T
import numpy as np
import nodes
import nodes.layers as layers
import nodes.iaf as iaf
import utils.mathT as mathT
import utils.mathZ as mathZ
import utils.plotZ as plotZ

f32 = theano.config.floatX
DIM = 2
SAMPLING_E = int(1e+5)
LR = 0.1

# defining target distribution
logPZ = mathT.gaussInit([0.,0.], np.eye(2)/2., mean=True)

# defining normalising flow model
iafmodel = iaf.IafStack('IAF_simple',dim=DIM)
layer1 = iaf.IafLinear({'name':'iaf-1','dim':2,'bias':True,'initmethod':'identity','lr':LR})
layer1.setWeight(weights=np.array([[0.3,0.2],[0,0.2]]), bias=[0,0])

layer2 = iaf.IafTanh({'name':'tanh'})
layer3 = iaf.IafPermute({'name':'per','dim':2})

layer4 = iaf.IafLinear({'name':'iaf-2','dim':2,'bias':True,'initmethod':'identity','lr':LR})
layer4.setWeight(weights=np.array([[0.3,-0.2],[0,-0.2]]), bias=[0,0])


iafmodel.add( layer1 )
iafmodel.add( layer2 )
iafmodel.add( layer3 )
iafmodel.add( layer4 )

# reparameterisation
e = T.fmatrix()
z,interout = iafmodel.forward(e,interout=True)
jaco = iafmodel.costMCLogQZ(e)
# gradw, gradb = iafmodel.getGrads(jaco)

fz = theano.function([e], z)
fjaco = theano.function([e], jaco)
# fgw = theano.function([e], gradw)
# fgb = theano.function([e], gradb)


# data_e = np.array([[1,2],[3,4]],dtype=f32)
# data_jaco = fjaco(data_e)
# print(data_z)
# print(data_jaco)
# print(fgw(data_e))
# print(fgb(data_e))


data_e = iafmodel.priorGen(50000,dtype=f32)
data_z = fz(data_e)

plotZ.hist_2d(data_z[:,0],data_z[:,1])
# kldiv = jaco - logPZ(z)
# grads = iafmodel.getGrads(kldiv)
