import numpy
from scipy import io
def addBiases(matrix):
    return numpy.c_[matrix, numpy.ones((matrix.shape[0], 1))]

def getData(datasetNum):
    dataDict = io.loadmat('res/dataset%s.mat' % str(datasetNum))
    negativeMatrix = addBiases(dataDict['neg_examples_nobias'])
    positiveMatrix = addBiases(dataDict['pos_examples_nobias'])
    weights = numpy.array(dataDict['w_init']).reshape((1,3))[0]
    return negativeMatrix, positiveMatrix, weights

