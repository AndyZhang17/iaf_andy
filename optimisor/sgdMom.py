import numpy as np
import theano
import theano.tensor as T
import utils
import optimisor

def opt(params, cost, paramshapes, inputs, outputs, mom=0.9, givens=None):
    lr = inputs[0]
    grads = [T.grad(cost,p) for p in params]
    lastdelta = [utils.sharedf(np.zeros(shp),borrow=True,name='lastchange-'+p.name)
                  for shp,p in zip(paramshapes,params)]
    newdelta = [mom*lastd+lr*g for lastd,g in zip(lastdelta,grads)]
    newparams = [p-delta for p,delta in zip(params,newdelta)]
    updates = zip(lastdelta,newdelta) + zip(params,newparams)
    return optimisor.makeFunc(inputs=inputs,outputs=outputs,updates=updates,givens=givens)
