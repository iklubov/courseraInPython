from builtins import range

import numpy

from Neuron import Neuron, NeuronTypes
import MatReader


def ketchup():
    neuron = Neuron([50, 50, 50], [2, 5, 3], 0, 850, 1/35, NeuronTypes.LINEAR)
   # neuron.target = 850
    for i in range(100):
        print('RESULT STEP=%s WEIGHTS=%s DW=%s' % (i, neuron.weights, neuron.dw))
        newWeights = [x + y for x, y in zip(neuron.weights, neuron.dw)]
        neuron.weights = newWeights

#exercise one - we are starting
def linearModel():
    negMat, posMat, weights = MatReader.getData(1)
    neuron = Neuron()
    neuron.learningRate = 0.3
    neuron.weights = weights
    neuron.type = NeuronTypes.LINEAR
    #print('WEIGHTS', neuron.weights)
    numMistakes = 0
    deltaw = numpy.zeros((1, negMat.shape[1]))
    for i in range(negMat.shape[1]+1):
        neuron.inputs = negMat[i]
        if neuron.output > 0:
            numMistakes += 1
            print(neuron.dw, deltaw)
            deltaw += neuron.dw
        #print('NEG', negMat[i], neuron.output)
        neuron.inputs = posMat[i]
        if neuron.output < 1:
            numMistakes += 1
            deltaw += neuron.dw
        #print('POS', posMat[i], neuron.output)
    print('new weights', neuron.weights + deltaw)
    pass

#ketchup()
linearModel()

