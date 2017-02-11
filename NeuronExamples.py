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
    def updateWeighs():
        numMistakes = 0
        deltaw = numpy.zeros(negMat.shape[1])

        for i in range(negMat.shape[1]+1):
            neuron.inputs = negMat[i]
            if neuron.output > 0:
                numMistakes += 1

                deltaw += neuron.dw

            neuron.inputs = posMat[i]
            if neuron.output < 0:
                numMistakes += 1
                deltaw += neuron.dw
        print('new weights', neuron.weights + deltaw, 'num mistakes', numMistakes)
        neuron.weights = neuron.weights + deltaw
        return numMistakes
        pass
    numMistakes = updateWeighs()
    while numMistakes > 0:
        numMistakes = updateWeighs()

#ketchup()
linearModel()

