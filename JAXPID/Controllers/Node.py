import numpy as np
import jax.numpy as jnp

class Node:
    def __init__(self):
        self.Inputs = []
        self.Weights = []
        self.Bias = np.random.randn()
        self.Output = None

    def AddOutput(self, Output):
        self.Output = Output

    def AddInput(self, Node, Weight):
        self.Inputs.append(Node)
        self.Weights.append(Weight)

    def CalcOutput(self, Weights):
        WeightedSum = 0
        for i in range(len(self.Inputs)):
            WeightedSum += self.Inputs[i].Output * Weights[i]
        self.Output = self.ActivationFunction(WeightedSum)

    def ActivationFunction(self, x):
        return jnp.maximum(0, x)
