import numpy
from scipy import io
#for i in range(4):
dataDict = io.loadmat('res/dataset1.mat')
#for key, value in dataDict.items():
negativeMatrix = numpy.matrix(dataDict['neg_examples_nobias'])
ones = numpy.ones((negativeMatrix.shape[0],1))
#negativeMatrix += ones
positiveMatrix = numpy.matrix(dataDict['pos_examples_nobias'])
weights = numpy.matrix(dataDict['w_init'])
print('dataDict', negativeMatrix, ones, numpy.c_[negativeMatrix, ones]) #negativeMatrix, positiveMatrix)

