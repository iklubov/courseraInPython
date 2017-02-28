from builtins import dict as dictdict

import numpy
import time

from numpy.core.multiarray import ndarray
from scipy import io
def addBiases(matrix):
    return numpy.c_[matrix, numpy.ones((matrix.shape[0], 1))]

def getData1(datasetNum):
    dataDict = io.loadmat('res/dataset%s.mat' % str(datasetNum))
    negativeMatrix = addBiases(dataDict['neg_examples_nobias'])
    positiveMatrix = addBiases(dataDict['pos_examples_nobias'])
    weights = numpy.array(dataDict['w_init']).reshape((1,3))[0]
    return negativeMatrix, positiveMatrix, weights

def getData2(minibatchSize=100):
    dataDict = io.loadmat('res/library.mat')['data']
    start_time = time.time()
    vocab = numpy.matrix(dataDict['vocab'][0][0][0])
    #print('VOCAB DATA INPUT', vocab)

    valid = getMatrix(dataDict, 'validData')
    valid_t = valid[0:len(valid)-1]
    valid_x = valid[len(valid)-1]
    #print('VALID DATA INPUT', valid_t, 'OUTPUT', valid_x)

    test = getMatrix(dataDict, 'testData')
    test_t = test[0:len(test) - 1]
    test_x = test[len(test) - 1]
    #print('TEST DATA INPUT', test_t, 'OUTPUT', test_x)

    train = getMatrix(dataDict, 'trainData')
    train_t = getBatchedData(train[0:len(train) - 1], minibatchSize)
    train_x = getBatchedData(train[len(train) - 1], minibatchSize)
    #print('TRAIN DATA INPUT', train_t.shape, 'OUTPUT', train_x.shape)


    elapsed_time = time.time() - start_time
    print('ELAPSED', elapsed_time)

def getMatrix(dataDict,dataField):
    rawdata = dataDict[dataField][0][0]

    #print('getMatrix', rawdata[0], len(rawdata), [r for r in rawdata])
    return rawdata

def getBatchedData(data, batchSize=1):
    dim = 1 if len(data.shape) == 1 else data.shape[0],  data.shape[0] if len(data.shape) == 1 else data.shape[1]
    result = numpy.ndarray((int(dim[1]/batchSize),), ndarray)
    for n in range(0, dim[1], batchSize):
        if dim[0] > 1:
            minibatch = data[:, n:(n+batchSize)]
        else:
            minibatch = data[n:(n + batchSize)]
        if len(minibatch) == batchSize:
            result[int(n/batchSize)] = minibatch
    return result

def printFields(*data):
    for d in data:
        print('FIELDS', isinstance(data, (dict)))#, [k for k in d.keys()] if isinstance(data, dict) else [k.size for k in data])
