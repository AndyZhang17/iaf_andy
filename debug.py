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

f32 = theano.config.floatX
DIM = 2
LR = 0.1

e = T.fmatrix()

# model defining
iafmodel = iaf.IafStack( 'IAF_MAP', dim=DIM )
layer1 = iaf.IafLinear('iaf1', DIM, lr=LR)
layerPer1 = iaf.IafPermute('per1', DIM)
layer2 = iaf.IafLinear('iaf2', DIM, lr=LR)


# w1 = np.array([[ 5.81540108,  0.9072175 ],[-0.,         -0.99228317]])
# b1 = np.array( [-7.87969303,  0.113997  ] )
# u1 = np.array( [ 3.64952326,  0.64334512] )
# w2 = np.array([[ 0.41677463,  0.26881257],[ 0.,          0.41814184]])
# b2 = np.array( [ 1.11006463, -0.6009872 ] )
# u2 = np.array( [-1.13209558, -1.79898369] )
# point_e = np.array([[-6.73204947, -1.90301168]], dtype=f32)




w1 = np.array( [[ 10.81296253,  0.28379461],[  0.        ,  -0.56237745]] )
b1 = np.array( [-9.890378,   -3.62310934] )
u1 = np.array( [  5.58141899e+00,  1.13344193e-03])
w2 = np.array( [[ 0.41775197, -2.56453133],[ 0.,         7.39298058]])
b2 = np.array( [-0.02870637, -2.38826609])
u2 = np.array( [-1.14617682, -2.8435452 ])
point_e = np.array( [[-1.15331483, -0.78353518]], dtype=f32)





layer1.setParams(w=w1, b=b1, u=u1)
layer2.setParams(w=w2, b=b2, u=u2)

iafmodel.add(layer1)
iafmodel.add(layerPer1)
iafmodel.add(layer2)

z,interz = iafmodel.forward(e, interout=True)

musin, varsin = [[0,0],[-2,1]], [np.eye(DIM)/5.,np.eye(DIM)/5.]
probs = [0.5,0.5]
logPZ = mathT.gaussMixInit(musin,varsin,probs,mean=True )
_,targetGen = mathZ.gaussMixInit(musin,varsin,probs)

# cost defining
logqz = iafmodel.mcLogQZ(e)
logpz = logPZ(z)
kldiv = logqz - logpz
iafmodel.setCost(kldiv)
updates = iafmodel.getUpdates()
logjaco1 = iafmodel.layers[0].logDetJaco()
logjaco2 = iafmodel.layers[2].logDetJaco()

# parameter transferring
# train = theano.function([e], kldiv, updates=updates)


fz      = theano.function([e],z)
fzinter = theano.function([e], interz)
fjaco1 = theano.function([e], logjaco1)
fjaco2 = theano.function([e], logjaco2)

fgrads  = theano.function([e], iafmodel.getGrads() )
fgradq  = theano.function([e], iafmodel.getGrads(logqz))
fgradp  = theano.function([e], iafmodel.getGrads(logpz))
fjaco1gradq = theano.function([e], iafmodel.layers[0].getGrads(logjaco1))
fjaco2gradq = theano.function([e], iafmodel.getGrads(logjaco2) )


print('> Function Construction done')

def showValues(values,name, items=None):
    print('\n> show: '+name)
    for i, v in enumerate(values):
        if items is not None:
            print(items[i])
        print(v)
    print(' ')

grad_names = ['w1','b1','u1','w2','b2','u2']
values = iafmodel.getValueParams()
showValues(values,'params', grad_names)


print('e input  : %s' %(point_e))
print('z output : %s' %(fz(point_e)))
showValues(fzinter(point_e), 'inter outputs')
print('jaco1    : %s' %(fjaco1(point_e)))
print('jaco2    : %s' %(fjaco2(point_e)))

showValues(fgrads(point_e), 'grads', grad_names)
# showValues(fgradq(point_e), 'grad-Q', grad_names)
# gradp = fgradp(point_e)
# showValues(gradp, 'grad-P',grad_names)
showValues(fjaco1gradq(point_e),'grad-d(jaco1)/d layer1',grad_names)
showValues(fjaco2gradq(point_e),'grad-d(jaco2)/d layer1&2',grad_names)




