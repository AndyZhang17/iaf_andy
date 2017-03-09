"""
Comparing adagrad, adadelta and constant learning in gradient descent(the seddle point function y^2 - x^2)
Reference:
1. comparison on several learning rate update scheme: http://ml.memect.com/archive/2014-12-12/short.html#3786866375172817
2. Saddle point, http://en.wikipedia.org/wiki/Saddle_point
"""
import numpy as np
import theano
import theano.tensor as T
import utils
import optimisor

def makeFunc(params, cost, updates):
    f = theano.function(
        inputs = [],
        outputs = cost,
        updates = updates
    )
    return f

## ------- ##
#  ADAGRAD  #
## ------- ##
def opt(params, cost, paramshapes, inputs,outputs,givens=None,gamma=.1, epsilon=.00001):
    '''
    :param params:
    :param cost:
    :param paramshapes:
    :param gamma:
    :param epsilon:
    :return: training function to be called
    '''
    grads = [T.grad(cost,param) for param in params]

    ghists = [ utils.sharedf(np.zeros(shape),borrow=True,name='gradhist:'+param.name)
                  for shape,param in zip(paramshapes,params) ]
    # newghists = [ ghist+T.sqr(g) for ghist,g in zip(ghists,grads) ]

    upt_ghists = [ (h,h+T.sqr(g)) for h,g in zip(ghists,grads) ]
        # zip(ghists,newghists)
    upt_params = [(p, p-(gamma*epsilon/(T.sqrt(gh)+epsilon))*g) for p,g,gh in zip(params,grads,ghists)]
    updates = upt_ghists+upt_params
    f = optimisor.makeFunc(inputs=inputs,outputs=outputs,updates=updates,givens=givens)
    return f



