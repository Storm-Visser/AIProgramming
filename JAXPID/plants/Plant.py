from random import seed
from random import uniform

# Abstract
class Plant:
    def __init__(self, TimestepsPerEpoch, NoiseRange):
        self.TimestepsPerEpoch = TimestepsPerEpoch
        self.NoiseRange = NoiseRange

    def Run(self, ControllerInput):
        Total = 0
        # run x amount of times
        for _ in range(self.TimestepsPerEpoch):
            Total = Total + self.CalcNewValues(ControllerInput)
        #Take the avg and return
        return Total/self.TimestepsPerEpoch
    
    #abstract
    def CalcNewValues(self, ControllerInput):
        return [0,0,0]

    def GenerateNoise(self):
        seed()
        Noise = uniform(self.NoiseRange[0], self.NoiseRange[1])
        return Noise