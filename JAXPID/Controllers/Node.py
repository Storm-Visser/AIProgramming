import numpy as np

class Node:
    def __init__(self):
        self.inputs = []
        self.weights = []
        self.bias = np.random.randn()
        self.output = None

    def AddOutput(self, output):
        self.output = output

    def AddInput(self, node, weight):
        self.inputs.append(node)
        self.weights.append(weight)

    def CalcOutput(self):
        weighted_sum = 0
        for i in range(len(self.inputs)):
            weighted_sum += self.inputs[i].output * self.weights[i]
        self.output = self.ActivationFunction(weighted_sum)

    def ActivationFunction(self, x):
        return 1 / (1 + np.exp(-x))