import theano

def makeFunc(inputs, outputs, updates, givens):
    f = theano.function(inputs=inputs,outputs=outputs, updates = updates, givens=givens)
    return f

import adagrad as adagrad
import sgd as sgd
import sgdMom as sgdMom
