import numpy as np
import jax
import jax.numpy as jnp

class Controller:

    InputValue = 0.0
    ErrorRate = 0.0
    K1 = 0
    K2 = 0
    K3 = 0

    def __init__(self, LearningRate):
        self.LearningRate = LearningRate

    def Analyse(self, Plant, ErrorRate):
        NewInputValue = 0.0
        #Update PID params
        self.K1 = self.K1 + (-1 * self.LearningRate * self.CalcJaxDerivative(Plant))
        #self.K2 = self.K2 + (-1 * self.LearningRate * self.CalcJaxDerivative(self.CalcErrorRateOverTime(ErrorRate), self.K1))
        #self.K3 = self.K3 + (-1 * self.LearningRate * self.CalcJaxDerivative(self.CalcErrorRateDerivitive(ErrorRate), self.K1))
        #update their values
        NewInputValue =  (self.K1 * ErrorRate)# + (self.K2 * self.CalcErrorRateOverTime(ErrorRate)) + (self.K3 * self.CalcErrorRateDerivitive(ErrorRate))
        self.InputValue = NewInputValue
        return NewInputValue



    def CalcErrorRateOverTime(self, ErrorRate):
        return 0

    def CalcErrorRateDerivitive(self, ErrorRate):
        return 0

    def CalcJaxDerivative(self, toDiff):
        result = jax.grad(toDiff)(self.InputValue)
        print("jax: " + str(result))
        return result

