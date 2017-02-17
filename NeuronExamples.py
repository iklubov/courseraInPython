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
class LinearModel():
    def __init__(self):
        negMat, posMat, weights = MatReader.getData1(1)
        self.neuron = Neuron(learningRate=0.2, weights=weights, type=NeuronTypes.LINEAR)
        self.numMistakes, self.deltaWeights = 1, 1
        while self.numMistakes > 0:
            self.countSystem(negMat, posMat, self.neuron)
            self.neuron.learningRate += 0.1
            self.deltaWeights = 1
            print('Mistakes', self.numMistakes)

    def updateWeighs(self, negMat, posMat, neuron):
        deltaw = numpy.zeros(negMat.shape[1])
        mistakes = 0
        for i in range(negMat.shape[1]+1):
            neuron.inputs = negMat[i]
            if neuron.output > 0:
                mistakes += 1

                deltaw += neuron.dw

            neuron.inputs = posMat[i]
            if neuron.output < 0:
                mistakes += 1
                deltaw += neuron.dw
        print('new weights', neuron.weights + deltaw, 'num mistakes', self.numMistakes, 'learningRate', self.neuron.learningRate)
        neuron.weights = neuron.weights + deltaw
        return mistakes

    def countSystem(self, *args):
        while self.deltaWeights > 0:
            lastWeights = self.neuron.weights.copy()
            self.numMistakes = self.updateWeighs(*args)
            self.deltaWeights = numpy.linalg.norm(lastWeights - self.neuron.weights)
            print('delta weights', numpy.linalg.norm(lastWeights - self.neuron.weights))



import matplotlib.pyplot as plt
def plot(data):
    plt.plot(data, 'bo')
    plt.ylabel('some numbers')
    plt.show()

#ketchup()
#linearModel = LinearModel()

def library():
    MatReader.getData2()

library()

