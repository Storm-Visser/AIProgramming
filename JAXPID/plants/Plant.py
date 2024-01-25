from random import seed
from random import uniform

# Abstract
class Plant:

    ErrorRateSum = 0.0
    PrevErrorRate = 0.0
    ErrorRate = 0.0

    def __init__(self, TimestepsPerEpoch, NoiseRange):
        self.TimestepsPerEpoch = TimestepsPerEpoch
        self.NoiseRange = NoiseRange


    def Run(self, K1, K2, K3):
        ControllerInput = (K1 * self.ErrorRate) + (K2 * (self.PrevErrorRate - self.ErrorRate)) + (K3 * self.ErrorRateSum)
        ER, RS = self.CalcNewValues(ControllerInput)
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