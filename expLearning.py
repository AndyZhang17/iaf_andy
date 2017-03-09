import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import numpy as np
import utils
import utils.plotZ as plotZ
import theano
import theano.tensor as T
import nodes.iaf2 as iaf
import model.mars1 as M
from optimisor.adagrad import opt as adagrad
from optimisor.sgd import opt as sgd

f32 = theano.config.floatX
SAMPLING = int(1e+1)
DECAY = 3e-5
LR0 = 0.01
MAXITER = 10000
VISUAL = True
LOG = './tmp/debug.npz'
utils.checkDir('./tmp', build=True)


print('... constructing model')
VALX = -.8
x = utils.sharedf([VALX,VALX])
model = M.banana()


print('... constructing variational distribution and symbol flows')
lr = T.fscalar()
qiaf = iaf.experiment(lr,2)
e = qiaf.getNoise(SAMPLING)
z = qiaf.forward(e, interout=False)

logqz = qiaf.mcLogQZ(e)
logpz = model.logPrior(z)
logpxz = model.logLik(x,z)
negELBO = -logpxz+logqz-logpz



print('... optimisor construct')
eStep = adagrad(params=qiaf.getParams(),cost=negELBO,
                paramshapes=qiaf.getParamShapes(),inputs=[],outputs=[negELBO])
# eStep = sgd(qiaf.getParams(),negELBO,inputs=[lr],outputs=[negELBO])


print('> Start optimisation')
record = {'ms':np.zeros((MAXITER,2)),'zs':np.zeros((MAXITER,SAMPLING,2))}

for i in range(MAXITER):
    LR = LR0/(1.+DECAY*i)
    # [cost] = eStep(LR)
    [cost] = eStep()
    record['ms'][i,0] = cost
    # record['zs'][i,:,:] = zsamples
    print('iter %d : ELBO %.4f' %(i,-cost))
    if np.isnan(cost):
        break
# np.savez(LOG,elbo=record['ms'][:,0],zsamples=record['zs'][:,:,:],valx=VALX)


if VISUAL:
    e1  = qiaf.getNoise(int(5e+4))
    z1  = qiaf.forward(e1,interout=False)
    fz1 = theano.function([],z1)
    data_z = fz1()
    plotZ.hist2D(data_z[:,:2], title='optimised posterior q( z | x=%.2f), iter %d'%(VALX,MAXITER))

plotZ.line_2d(record['ms'][:,0]*(-1.),title='ELBO')

def showValues(values, name):
    print('\n> '+name)
    for v in values:
        print('>>\n%s'%v)
    print(' ')
values = qiaf.getValueParams()
showValues(values, 'trained parameters')
