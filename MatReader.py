from builtins import dict as dictdict

import numpy
import time
from scipy import io
def addBiases(matrix):
    return numpy.c_[matrix, numpy.ones((matrix.shape[0], 1))]

def getData1(datasetNum):
    dataDict = io.loadmat('res/dataset%s.mat' % str(datasetNum))
    negativeMatrix = addBiases(dataDict['neg_examples_nobias'])
    positiveMatrix = addBiases(dataDict['pos_examples_nobias'])
    weights = numpy.array(dataDict['w_init']).reshape((1,3))[0]
    return negativeMatrix, positiveMatrix, weights

def getData2():
    dataDict = io.loadmat('res/library.mat')['data']
    vocab = numpy.matrix(dataDict['trainData'][0][0][0])
    train = getMatrix(dataDict, 'trainData')
    train_x = train[3]
   # validData = numpy.matrix(dataDict['validData'][0][0])
    testData = dataDict['testData'][0][0]
    start_time = time.time()
    print('T',train_x[0][0][0])
    #tranReverse = [(vocab[i][j] for j in range(vocab.shape[0])) for i in range(vocab.shape[1])]
   # print('TRANSPOSE', vocab)
    elapsed_time = time.time() - start_time
   # print('EX2 GET DATA', elapsed_time)
    #return negativeMatrix, positiveMatrix, weights

def getMatrix(dataDict,dataField):
    print('getMatrix', numpy.matrix(dataDict[dataField][0][0]))#[0][0].shape)
    return numpy.matrix(dataDict[dataField][0][0])

def printFields(*data):
    for d in data:
        print('FIELDS', isinstance(data, (dict)))#, [k for k in d.keys()] if isinstance(data, dict) else [k.size for k in data])
