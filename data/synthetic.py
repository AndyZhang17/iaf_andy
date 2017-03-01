import data as D
import utils.mathZ as M


def gaussSingle(muin,varin,size,name='gaussian_single'):
    data = D.dataset(name)
    logP, gen = M.gaussInit(muin,varin)
    data.setTrain( gen(size) )
    return data


# def gaussMix(size,shape,)

