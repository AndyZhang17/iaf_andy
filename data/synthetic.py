import data as D
import utils
import utils.mathZ as mathZ
import numpy as np

def genGMM( numt,musin,varsin,probs,filepath):
    logP, gen = mathZ.gaussMixInit(musin,varsin,probs)
    train = gen(numt)
    np.savez(filepath,x=train)


utils.checkDir('./tmp',build=True)
musin = [[0,0],[-2,1]]
varsin = [np.eye(2)/5., np.eye(2)/3]
probs = [0.5,0.5]
genGMM( 1000, musin, varsin, probs, '/tmp/gmm1.npz' )





