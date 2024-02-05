import numpy as np
import jax
import jax.numpy as jnp

class Controller:

    

    def __init__(self, LearningRate):
        self.LearningRate = LearningRate
        self.InputValue = [0.0,0.0,0.0]
        self.K1 = 0.0 # error rate
        self.K2 = 0.0 # error rate prev/error rate
        self.K3 = 0.0 # error rate sum

    def Analyse(self, Plant):
        #get the new input for the plant
        NewInputValue = [0.0,0.0,0.0]
        DF1 = jax.grad(lambda x,y,z: Plant(x,y,z)[0], argnums=0)
        DF2 = jax.grad(lambda x,y,z: Plant(x,y,z)[0], argnums=1)
        DF3 = jax.grad(lambda x,y,z: Plant(x,y,z)[0], argnums=2)
        #Update PID params
        self.K1 = self.K1 + (-1 * self.LearningRate * DF1(self.InputValue[0], self.InputValue[1], self.InputValue[2]))
        self.K2 = self.K2 + (-1 * self.LearningRate * DF2(self.InputValue[0], self.InputValue[1], self.InputValue[2]))
        self.K3 = self.K3 + (-1 * self.LearningRate * DF3(self.InputValue[0], self.InputValue[1], self.InputValue[2]))
        #pass the new values
        NewInputValues =  [self.K1, self.K2, self.K3]
        self.InputValue = NewInputValue
        return NewInputValues

