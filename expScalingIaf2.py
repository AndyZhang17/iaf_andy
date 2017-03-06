import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import theano
import theano.tensor as T
import numpy as np
import nodes
import nodes.iaf2 as iaf
import utils.mathT as mathT
import utils.mathZ as mathZ
import utils.plotZ as plotZ
import utils.spec as spec
import time

f32 = theano.config.floatX
DIM = 2
SAMPLING = int(3e+2)
SAMPLING_SHOW = int(5e+4)

DECAY = 1e-3
LR0   = 0.1
MAXITER = 1000


## -------- ##
#   TARGET   #
## -------- ##
# logPZ = mathT.gaussInit([-2,1],np.eye(DIM),mean=True)
# musin, varsin, probs = [[0,0],[-2,1],[1,+2]], [np.eye(DIM)/5.,np.eye(DIM)/5.,np.eye(DIM)/3.], [0.4,0.3,0.3]
# musin, varsin, probs = [[0,0],[-2,1]], [np.eye(DIM)/5.,np.eye(DIM)/5.], [0.5,0.5]
musin, varsin, probs = spec.gmmParam(2,dim=DIM)
logPZ = mathT.gaussMixInit(musin,varsin,probs,mean=True )
_,targetGen = mathZ.gaussMixInit(musin,varsin,probs)
plotZ.hist2D(targetGen(SAMPLING_SHOW))


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
layerPer4 = iaf.IafPermute('per4', DIM)
layer5 = iaf.IafLinear('iaf5', DIM, lr=lr)

iafmodel.add([layer1, layerPer1])
iafmodel.add([layer2, layerPer2])
iafmodel.add([layer3, layerPer3])
iafmodel.add([layer4, layerPer4])
iafmodel.add(layer5)


## ----------------------- ##
#  Symbolic initialisation  #
## ----------------------- ##
print('... symbolic initialisation')
e = iafmodel.getNoise(SAMPLING)
z,interz = iafmodel.forward(e, interout=True)

logqz = iafmodel.mcLogQZ(e)
logpz = logPZ(z)
kldiv = logqz - logpz

iafmodel.setCost(kldiv)
updates = iafmodel.getUpdates()

print('... function construct')
train = theano.function([lr], kldiv, updates=updates )
fz = theano.function([],z)

print('> Start training, maxiter %d' % (MAXITER))


def showValues(values, name):
    print('\n> '+name)
    for v in values:
        print('>>\n%s'%v)
    print(' ')

def report(iter, cost, logq, logp, grads):
    '''inputs are all numpy memories'''
    numG = len( grads )/3
    outs = 'iter %s: KL %.4f | %.2f - %.2f | '
    outg = '|'.join([ ' Grad'+str(i+1)+': %.2f %.2f %.2f ' for i in range(numG) ])
    nums = [iter,cost,logq,logp]
    nums.extend(grads)
    out = outs+outg
    print(out % tuple(nums))


KL_list = list()


## RECORDING
i = 0
t0 = time.time()
while i<MAXITER:
    i+=1
    LR = LR0/(1+DECAY*i)
    cost = train(LR)
    KL_list.append(cost)
    print('iter %d : KL %.4f' % (i,cost))

tused = time.time()-t0
print('\n> total time used: %.2f min, per-iteration: %.2f s' %(tused/60, tused/float(i)))


data_z = fz()
plotZ.hist2D(data_z)
plotZ.line_2d(KL_list)
