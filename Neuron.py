import math
from builtins import range


class Utils:
    @staticmethod
    def multiplyVectors(v1, v2):
        assert len(v1) == len(v2)
        result = 0
        for i in range(len(v1)):
            result += v1[i]*v2[i]
        return result

    @staticmethod
    def logistic(input):
        return 1 / (1+math.exp(-input))

class NeuronTypes:
    LINEAR, LOGISTIC, BINARYTHRESHOLD = range(3)

class Neuron:
    def __init__(self, weights=[], inputs=[], bias=0, target=0, learningRate=0, type=NeuronTypes.LOGISTIC):
        self.__weights = weights
        self.bias = bias
        self.type = type
        self.__inputs = inputs if len(inputs) else [0] * len(weights)
        self.__dw =     [0] * len(weights)
        self.__target = target
        self.__learningRate = learningRate
        self.__updateOutput()
        self.__updateDW()

    @property
    def inputs(self):
        return self.__inputs

    @inputs.setter
    def inputs(self, value):
        self.__inputs = value
        self.__updateOutput()
        self.__updateDW()

    @property
    def output(self):
        return self.__output

    @property
    def dw(self):
        return self.__dw

    @property
    def learningRate(self):
        return self.__learningRate

    @learningRate.setter
    def learningRate(self, value):
        self.__learningRate = value
        self.__updateDW()

    @property
    def target(self):
        return self.__target

    @target.setter
    def target(self, value):
        self.__target = value
        self.__updateDW()

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, value):
        self.__weights = value
        if len(self.__inputs) == len(value):
            self.__updateOutput()
            self.__updateDW()

    def derivative(self):
        linearInput = self.bias + Utils.multiplyVectors(self.__inputs, self.__weights)
        if self.type == NeuronTypes.LINEAR:
            return 0
        elif self.type == NeuronTypes.LOGISTIC:
            return Utils.logistic(linearInput)(1-Utils.logistic(linearInput))
        elif self.type == NeuronTypes.BINARYTHRESHOLD:
            return 0


    def __updateOutput(self):
        linearInput = self.bias + Utils.multiplyVectors(self.__inputs, self.__weights)
        if self.type == NeuronTypes.LINEAR:
            self.__output = linearInput
        elif self.type == NeuronTypes.LOGISTIC:
            self.__output = 1 / (1+math.exp(-linearInput))
        elif self.type == NeuronTypes.BINARYTHRESHOLD:
            self.__output = 0 if linearInput < 0 else 1

    def __updateDW(self):
        if len(self.__dw) == 0:
            self.__dw = [0]*len(self.__weights)
        if self.type == NeuronTypes.LINEAR:
            for i in range(len(self.__weights)-1):
                self.__dw[i] = self.learningRate * self.__inputs[i] * (self.target - self.__output)




