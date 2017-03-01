import theano
import theano.tensor as T
import numpy as np
import nodes
import nodes.layers as layers
import nodes.iaf as iaf
import utils.mathT as mathT
import utils.mathZ as mathZ
import utils.plotZ as plotZ

f32 = 'float32'
DIM = 2
SAMPLING_E = int(1e+5)
LR = 0.1
iafmodel = iaf.IafStack('IAF_simple',dim=DIM)
# iafmodel.add( iaf.IafLinear({'name':'iaf-1','dim':DIM,'bias':True,'initmethod':'identity','lr':LR}) )
# iafmodel.add( iaf.IafTanh({'name':'tanh'}) )
# iafmodel.add( iaf.IafPermute({'name':'permute','dim':DIM}) )
# iafmodel.add( iaf.IafLinear({'name':'iaf-2','dim':DIM,'bias':True,'initmethod':'identity','lr':LR}) )
# iafmodel.add( iaf.IafTanh({'name':'tanh'}) )
# iafmodel.add( iaf.IafPermute({'name':'permute','dim':DIM}) )
# iafmodel.add( iaf.IafLinear({'name':'iaf-3','dim':DIM,'bias':True,'initmethod':'identity','lr':LR}) )
# iafmodel.add( iaf.IafTanh({'name':'tanh'}) )
# iafmodel.add( iaf.IafPermute({'name':'permute','dim':DIM}) )


# iafmodel.layers[2].setWeight(np.array([[0,1],[1,0]],dtype=f32))

iafmodel.add(iaf.IafNormalising({'name':'iaf-1','dim':DIM,'lr':LR}))
iafmodel.add(iaf.IafNormalising({'name':'iaf-1','dim':DIM,'lr':LR}))
iafmodel.add(iaf.IafNormalising({'name':'iaf-1','dim':DIM,'lr':LR}))

e = T.fmatrix()

# reparameterisation
z,interout = iafmodel.forward(e,interout=True)
fz  = theano.function([e],z)

musin = [[0,0],[-1,1]]
varsin = [np.eye(DIM)/10.,np.eye(DIM)/10.]
probs = [0.5,0.5]
logPZ = mathT.gaussMixInit(musin,varsin,probs,mean=True )
_,targetGen = mathZ.gaussMixInit(musin,varsin,probs)

data_e = iafmodel.priorGen(100,dtype=f32)

# import nodes as N
# import theano.tensor.nnet as nnet
# import numpy.random as npr
# w = N.sharedf([0.]*DIM)
# b = N.sharedf([0.]*DIM)
# u = N.sharedf( ((npr.rand(DIM)>0.5)-0.5)/50. )
# act = T.dot(e,w)+b
# ftmp = theano.function([e],act)
# print(np.shape(ftmp(data_e)))


print(np.shape(data_e))
data_z = fz(data_e)
print(np.shape(data_z))

kldiv = iafmodel.costMCLogQZ(e) - logPZ(z)
fkl = theano.function([e], kldiv, updates=iafmodel.getUpdates(kldiv))

print('> Function Construction done')

print('----')
for i in range(100):
    # if i==100:
    #     for layer in iafmodel.layers:
    #         layer.lr = LR2
    #         fkl = theano.function([e], kldiv, updates=iafmodel.getUpdates(kldiv))
    data_e = iafmodel.priorGen(SAMPLING_E,dtype=f32)
    curcost = fkl(data_e)
    print('iter %s : KL-div %.4f' %(i,curcost))
print('----')

params = iafmodel.showParam()
for param in params:
    print(param)

z = fz(data_e)
tarz = targetGen(SAMPLING_E)
plotZ.hist_2d(tarz[:,0],tarz[:,1])
plotZ.hist_2d(z[:,0],z[:,1])


