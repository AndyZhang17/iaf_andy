import theano
import theano.tensor as T
import numpy as np
import nodes
import nodes.layers as layers
import nodes.iaf as iaf


iafmodel = iaf.IafStack('IAF_simple')
iafmodel.add( iaf.IafLinear({'name':'iaf-1','dim':3,'bias':True,'initmethod':'identity'}) )
iafmodel.add( iaf.IafLinear({'name':'iaf-2','dim':3,'bias':True,'initmethod':'test2'}) )
iafmodel.add( iaf.IafSigmoid({'name':'sig-2'}))

x = T.fmatrix()
y = iafmodel.forward(x)
totalJaco = iafmodel.logDetJacoSum()

fx = theano.function([x],y)
jaco = theano.function([x],totalJaco)


input = np.array([[1,2,3],[4,5,6]],dtype='float32')
print(fx(input))
print(jaco(input))

jaco0 = iafmodel.layers[0].logDetJacobian()
jaco1 = iafmodel.layers[1].logDetJacobian()
jaco2 = iafmodel.layers[2].logDetJacobian()

fj0 = theano.function([],jaco0)
fj1 = theano.function([],jaco1)
fj2 = theano.function([x],jaco2)
print
print(fj0())
print(fj1())
print(fj2(input))