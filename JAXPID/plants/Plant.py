from random import seed
from random import uniform

# Abstract
class Plant:
    def __init__(self, TimestepsPerEpoch, NoiseRange):
        self.TimestepsPerEpoch = TimestepsPerEpoch
        self.NoiseRange = NoiseRange

    def Run(self, ControllerInput):
        ER, RS = self.CalcNewValues(ControllerInput)
        return ER, RS
    
    def GenerateNoise(self):
        seed()
        Noise = uniform(self.NoiseRange[0], self.NoiseRange[1])
        return Noise

    #abstract methods
    def CalcNewValues(self, ControllerInput):
        return 0

    def Update():
        return 0 