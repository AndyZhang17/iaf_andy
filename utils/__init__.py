import numpy as np
import numpy.random as npr
import mathz as mathz


def checkDictKeys(dictin,members,rerr=False):
    valid = np.prod(np.array([mem in dictin.keys() for mem in members])+0)
    if rerr and (not valid):
        raise ValueError('Input dict doesn\'t satisfy:\ndict: %s\nrequired: %s' % (repr(dictin),repr(members)) )
    return valid
