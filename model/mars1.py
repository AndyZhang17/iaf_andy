import model as model
import theano.tensor as T
import theano.tensor.nlinalg as nlinalg
import utils.mathT as mathT
import utils
import numpy as np

class banana(model.model):
    '''
    x = z1*z2 + noise
    '''
    def __init__(self):
        '''
        :return:
        '''
        super(banana,self).__init__()
        self.priormus = utils.sharedf(np.zeros(2))
        self.priorvar = utils.sharedf(np.eye(2))

        self.stdn = utils.sharedf([.5,.5])
        self.varn = nlinalg.diag(T.sqr(self.stdn))

        self.logPz = mathT.gaussInit(self.priormus,self.priorvar,mean=True)

    def logPrior(self,z):
        return self.logPz(z)

    def logLik(self,x,z):
        '''
        model specific likelihood
        :return: log-P(x|z)
        '''
        logPxz = mathT.gaussInit(x,self.varn,mean=True)
        z1z2 = T.prod(z,axis=1,keepdims=True)
        return logPxz(T.concatenate([z1z2,z1z2],axis=1))






