__author__ = 'andy17'
import numpy as np
class dataset(object):
    def __init__(self,name):
        self.name=name
        self.train = None
        self.test  = None
        self.valid = None

    def setTrain(self,train):
        self.train = train
        self.shape = np.shape(train[0])
        self.numtrain = len(self.train)


