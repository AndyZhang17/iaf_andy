import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import theano
import theano.tensor as T
# from theano.compile.nanguardmode import NanGuardMode
import numpy as np
import nodes
import nodes.iaf2 as iaf
import utils.mathT as mathT
import utils.mathZ as mathZ
import utils.plotZ as plotZ
import utils.spec as spec
import time

f32 = theano.config.floatX
DIM = 8
SAMPLING = int(1e+1)
SAMPLING_SHOW = int(5e+4)

DECAY = 1e-3
LR0   = 0.1
MAXITER = 100


## -------- ##
#   TARGET   #
## -------- ##
# logPZ = mathT.gaussInit([-2,1],np.eye(DIM),mean=True)
# musin, varsin = [[0,0],[-2,1]], [np.eye(DIM)/5.,np.eye(DIM)/5.]
# probs = [0.5,0.5]
# musin, varsin = [[0,0],[-2,1],[1,+2]], [np.eye(DIM)/5.,np.eye(DIM)/5.,np.eye(DIM)/3.]
# probs = [0.4,0.3,0.3]
musin, varsin, probs = spec.gmmParam(1,dim=DIM)
logPZ = mathT.gaussMixInit(musin,varsin,probs,mean=True )
_,targetGen = mathZ.gaussMixInit(musin,varsin,probs)
# plotZ.hist2D(targetGen(SAMPLING))


## ----- ##
#  MODEL  #
## ----- ##
print('... constructing models')
e = T.fmatrix()
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

z,interz = iafmodel.forward(e, interout=True)

## ---- ##
#  COST  #
## ---- ##
logqz = iafmodel.mcLogQZ(e)
logpz = logPZ(z)
kldiv = logqz - logpz

iafmodel.setCost(kldiv)
updates = iafmodel.getUpdates()
grads = [ T.mean(g) for g in iafmodel.getGrads() ]


# parameter transferring
train = theano.function([e,lr], kldiv, updates=updates )
                # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
fz = theano.function([e],z)
fzinter = theano.function([e], interz)
fgrads = theano.function([e], [ T.mean(g) for g in iafmodel.getGrads() ] )
fgradq = theano.function([e], [ g for g in iafmodel.getGrads(logqz)])
fgradp = theano.function([e], [ g for g in iafmodel.getGrads(logpz)])
flogpz = theano.function([e], logpz )
flogqz = theano.function([e], logqz )

print('> Function Construction done')



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


values = iafmodel.getValueParams()
showValues(values, 'initial parameters')


## RECORDING
i = 0
t0 = time.time()
tlist = list()
kllist = list()
varlist = list()

while i<MAXITER:
    i+=1
    LR = LR0/(1+DECAY*i)
    data_e = iafmodel.eGen(SAMPLING,dtype=f32)

    data_z = fz(data_e)
    varlist.append( np.trace( np.dot(data_z.T,data_z)) /DIM/SAMPLING )

    curcost = train(data_e, LR)
    kllist.append(curcost)
    gs = fgrads(data_e)
    report(i,curcost,flogqz(data_e),flogpz(data_e),gs)



    if np.isnan( np.sum(gs) ):
        print('\n> catching nan: ')
        values = iafmodel.getValueParams()
        showValues(values, 'Parameters')
        numd, dim = np.shape(data_e)
        for idx in range(numd):
            point_e = np.array( [data_e[idx,:]] )
            gpoint = fgrads(point_e)
            gradq = fgradq(point_e)
            gradp = fgradp(point_e)
            if np.isnan(np.sum(gpoint)):
                print( '%s e : %s  \t z : %s' %( idx, point_e, fz(point_e)))
                showValues(fzinter(point_e), 'inter Outputs')
                print( '\tw\tb\tu\t\tw\tb\tu ')
                print( 'grads   : %s ' %(gpoint))
                print( 'grads_q : %s ' %(gradq))
                print( 'grads_p : %s ' %(gradp))
                values = iafmodel.getValueParams()
                break
        break

    # if i==int(MAXITER/4) or i==int(MAXITER/2) or i==int(MAXITER*3/4):
    #     tlist.append(time.time()-t0)
    #     t0 = time.time()
    #     data_e = iafmodel.eGen(SAMPLING,dtype=f32)
    #     data_z = fz(data_e)
    #     plotZ.hist2D(data_z)

tlist.append(time.time()-t0)
tused = np.sum( tlist )
print('\n> total time used: %.2f min' %(tused/60.))

# values = iafmodel.getValueParams()
# showValues(values)
data_e = iafmodel.eGen(SAMPLING_SHOW,dtype=f32)
data_z = fz(data_e)
# plotZ.hist2D(data_z)
title = 'KL vs iter, time used %.2f min' %(tused/60.)
# plotZ.line_2d(range(1,len(kllist)+1),kllist,xname='iterations',yname='KL',title=title)
plotZ.line_2d(range(1,len(varlist)+1),varlist,xname='iterations',yname='KL',title=title)