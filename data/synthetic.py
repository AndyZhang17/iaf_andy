import data as D
import utils
import utils.mathZ as mathZ
import numpy as np

def genGMM( numt,musin,varsin,probs,filepath):
    logP, gen = mathZ.gaussMixInit(musin,varsin,probs)
    train = gen(numt)
    np.savez(filepath,x=train)

def genGaus( num, muin, varin, filepath):
    logP, gen = mathZ.gaussInit(muin,varin)
    np.savez(filepath,x=gen(num))


utils.checkDir('./tmp',build=True)

# musin = [[0,0],[-2,1]]
# varsin = [np.eye(2)/5., np.eye(2)/5]
# probs = [0.5,0.5]
# genGMM( 2000, musin, varsin, probs, '/tmp/gmm1.npz' )

muin = [2,1]
varin = np.eye(2)/6.
genGaus(2000,muin,varin,'/tmp/gmm0.npz')



