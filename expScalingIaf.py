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

f32 = theano.config.floatX
DIM = 2
SAMPLING = int(5e+4)
LR = 0.1
MAXITER = 2000

e = T.fmatrix()

# model defining
iafmodel = iaf.IafStack( 'IAF_MAP', dim=DIM )
layer1 = iaf.IafLinear('iaf1', DIM, lr=LR)
layerPer1 = iaf.IafPermute('per1', DIM)
layer2 = iaf.IafLinear('iaf2', DIM, lr=LR)
layerPer2 = iaf.IafPermute('per2', DIM)
layer3 = iaf.IafLinear('iaf3', DIM, lr=LR)
layerPer3 = iaf.IafPermute('per3', DIM)
layer4 = iaf.IafLinear('iaf4', DIM, lr=LR)


iafmodel.add(layer1)
iafmodel.add(layerPer1)
iafmodel.add(layer2)
iafmodel.add(layerPer2)
iafmodel.add(layer3)
iafmodel.add(layerPer3)
iafmodel.add(layer4)

z,interz = iafmodel.forward(e, interout=True)

# target defining
# logPZ = mathT.gaussInit([-2,1],np.eye(DIM),mean=True)
musin, varsin = [[0,0],[-2,1]], [np.eye(DIM)/5.,np.eye(DIM)/5.]
probs = [0.5,0.5]
# musin, varsin = [[0,0],[-2,1],[1,+2]], [np.eye(DIM)/5.,np.eye(DIM)/5.,np.eye(DIM)/3.]
# probs = [0.4,0.3,0.3]

logPZ = mathT.gaussMixInit(musin,varsin,probs,mean=True )
_,targetGen = mathZ.gaussMixInit(musin,varsin,probs)

plotZ.hist2D(targetGen(SAMPLING))


# cost defining
logqz = iafmodel.mcLogQZ(e)
logpz = logPZ(z)
kldiv = logqz - logpz
# kldiv = iafmodel.mcLogQZ(e)-logPZ(z)
iafmodel.setCost(kldiv)
updates = iafmodel.getUpdates()
grads = [ T.mean(g) for g in iafmodel.getGrads() ]


# parameter transferring
train = theano.function([e], kldiv, updates=updates )
                # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
fz = theano.function([e],z)
fzinter = theano.function([e], interz)
fgrads = theano.function([e], [ T.mean(g) for g in iafmodel.getGrads() ] )
fgradq = theano.function([e], [ g for g in iafmodel.getGrads(logqz)])
fgradp = theano.function([e], [ g for g in iafmodel.getGrads(logpz)])

flogpz = theano.function([e],logPZ(z))
flogqz = theano.function([e], iafmodel.mcLogQZ(e))

print('> Function Construction done')

values = iafmodel.getValueParams()
def showValues(values, name):
    print('\n> '+name)
    for v in values:
        print(v)
    print(' ')


showValues(values, 'Params')
i,curcost = 0, 100.
while i<MAXITER:
    i+=1
    data_e = iafmodel.eGen(SAMPLING,dtype=f32)
    curcost, oldcost = train(data_e), curcost
    gs = fgrads(data_e)
    print('iter %s : KL-div %.4f |   %.2f - %.2f  | Grad1: w %.2f b %.2f u %.2f | Grad2: w %.2f b %.2f u %.2f | Grad3: w %.2f b %.2f u %.2f'\
          %(i,curcost, flogqz(data_e), flogpz(data_e), gs[0], gs[1], gs[2], gs[3], gs[4], gs[5], gs[6], gs[7], gs[8] ) )


    if np.isnan( np.sum(gs) ):
        print('> catching nan: ')
        values = iafmodel.getValueParams()
        showValues(values, 'Params')
        numd, dim = np.shape(data_e)
        for idx in range(numd):
            point_e = np.array( [data_e[idx,:]] )
            gpoint = fgrads(point_e)
            gradq = fgradq(point_e)
            gradp = fgradp(point_e)
            if np.isnan(np.sum(gpoint)):
                print( '%s e : %s  \t z : %s' %( idx, point_e, fz(point_e)))
                showValues(fzinter(point_e), 'interOut')
                print( '          w\tb\tu\t\tw\tb\tu ')
                print( 'grads   : %s ' %(gpoint))
                print( 'grads_q : %s ' %(gradq))
                print( 'grads_p : %s ' %(gradp))
                values = iafmodel.getValueParams()
                showValues(values,'Params')
                break
        break

    # if i==int(MAXITER/3) or i==int(2*MAXITER/3):
    #     data_e = iafmodel.eGen(SAMPLING,dtype=f32)
    #     data_z = fz(data_e)
    #     plotZ.hist2D(data_z)



# values = iafmodel.getValueParams()
# showValues(values)
data_e = iafmodel.eGen(SAMPLING,dtype=f32)
data_z = fz(data_e)
plotZ.hist2D(data_z)