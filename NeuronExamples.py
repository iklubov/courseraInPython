from builtins import range

from Neuron import Neuron, NeuronTypes
import MatReader


def ketchup():
    neuron = Neuron([50, 50, 50], [2, 5, 3], 0, 850, 1/35, NeuronTypes.LINEAR)
   # neuron.target = 850
    for i in range(100):
        print('RESULT STEP=%s WEIGHTS=%s DW=%s' % (i, neuron.weights, neuron.dw))
        newWeights = [x + y for x, y in zip(neuron.weights, neuron.dw)]
        neuron.weights = newWeights

def example1():
    pass

#ketchup()

