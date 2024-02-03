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

    def CalcOutput(self, Weights, ActivationFunction):
        WeightedSum = 0
        for i in range(len(self.Inputs)):
            WeightedSum += self.Inputs[i].Output * Weights[i]
        if ActivationFunction == 1:
            self.Output = self.ActivationFunctionSigmoid(WeightedSum)
        elif ActivationFunction == 2:
            self.Output = self.ActivationFunctionTanh(WeightedSum)
        elif ActivationFunction == 3:
            self.Output = self.ActivationFunctionRelu(WeightedSum)

    def ActivationFunctionSigmoid(self, x):
        return 1 / (1 + jnp.exp(-x))

    def ActivationFunctionTanh(self, x):
        return jnp.tanh(x)

    def ActivationFunctionRelu(self, x):
        return jnp.maximum(0, x)
