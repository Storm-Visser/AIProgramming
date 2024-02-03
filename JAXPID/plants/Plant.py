from random import seed
from random import uniform

# Abstract
class Plant:

    def __init__(self, TimestepsPerEpoch, NoiseRange):
        self.TimestepsPerEpoch = TimestepsPerEpoch
        self.NoiseRange = NoiseRange
        self.ErrorRateSum = 0.0
        self.PrevErrorRate = 0.0
        self.ErrorRate = 0.0


    def Run(self, K1, K2, K3):
        ControllerInput = (K1 * self.ErrorRate) + (K2 * (self.ErrorRate - self.PrevErrorRate)) + (K3 * self.ErrorRateSum)
        ER, RS = self.CalcNewValues(ControllerInput)
        return ER, RS

    def RunNN(self, Output):
        ER, RS = self.CalcNewValues(Output)
        return ER, RS
    
    def GenerateNoise(self):
        Noise = uniform(self.NoiseRange[0], self.NoiseRange[1])
        return Noise

    def UpdateErrorVars(self, errorRate):
        self.PrevErrorRate = self.ErrorRate
        self.ErrorRate = errorRate
        self.ErrorRateSum = self.ErrorRateSum + errorRate

    #abstract methods
    def CalcNewValues(self, ControllerInput):
        return 0

    def Update():
        return 0 