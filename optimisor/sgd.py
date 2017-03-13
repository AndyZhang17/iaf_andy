import numpy as np
import theano as theano
import theano.tensor as T
import utils
import optimisor

def opt(params,cost,inputs,outputs,consider_constant,givens=None):
    '''
    inputs all symbolic
    inputs[0] has to be the learning rate
    '''
    lr = inputs[0]
    grads = [T.grad(cost,param,consider_constant=consider_constant) for param in params]
    updates = [(p,p-lr*g) for p,g in zip(params,grads)]
    f = optimisor.makeFunc(inputs=inputs,outputs=outputs,updates=updates,givens=givens)
    return f