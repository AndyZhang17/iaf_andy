import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import numpy as np
import utils
import utils.plotZ as plotZ
import theano
import theano.tensor as T
import theano.tensor.nlinalg as nlinalg
import nodes.iaf2 as iaf
import model.mars1 as M
from optimisor.adagrad import opt as adagrad
from optimisor.sgd import opt as sgd
from optimisor.sgdMom import opt as sgdMom

f32 = theano.config.floatX
SAMPLING = int(10)
DECAY = 1e-3
LR0 = 0.05
MAXITER = 3000
VISUAL = True
utils.checkDir('./tmp', build=True)




print('... constructing model')
VALX = -1.2
x = utils.sharedf([VALX,VALX])
model = M.banana()


print('... variational distribution and symbol flows')
lr = T.fscalar()
qiaf = iaf.experiment(lr,2)
e = qiaf.getNoise(SAMPLING)
z = qiaf.forward(e, interout=False)

logqz = qiaf.mcLogQZ(e)
logpz = T.mean(model.logPrior(z))
logpxz = T.mean(model.logLik(x,z))
negELBO = -logpxz+logqz-logpz



print('... optimisor')
# eStep = adagrad(params=qiaf.getParams(),cost=negELBO,
#                 paramshapes=qiaf.getParamShapes(),inputs=[],outputs=[negELBO])
eStep = sgd(qiaf.getParams(),negELBO,
            inputs=[lr],outputs=[negELBO,z,logqz,logpz,logpxz,e],
            consider_constant=[])
# eStep = sgdMom(qiaf.getParams(),negELBO,qiaf.getParamShapes(),
#                inputs=[lr],outputs=[negELBO,z,logqz,logpz,logpxz,e],mom=0.)

print('> Start optimisation')
EXP_NAME = 'sgd1'
LOG = './tmp/debug_'+EXP_NAME+'.npz'
record = {'ms':np.zeros((MAXITER,4)),'zs':np.zeros((MAXITER,SAMPLING,2)),
          'es':np.zeros((MAXITER,SAMPLING,2)) }

for i in range(MAXITER):
    LR = LR0/(1.+DECAY*i)
    [cost,zs,qz,pz,pxz,es] = eStep(LR)
    # [cost] = eStep()
    record['ms'][i,:4] = [cost,qz,pz,pxz]
    record['zs'][i,:,:] = zs
    record['es'][i,:,:] = es
    print('iter %d : ELBO %.4f' %(i,-cost))
    if np.isnan(cost):
        break

print('... saving LOG to \'%s\''%(LOG))
np.savez(LOG,ms=record['ms'][:,:],
         zs=record['zs'][:,:,:],
         es = record['es'][:,:,:],
         valx=VALX)

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
