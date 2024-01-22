from random import seed
from random import randint

class Plant:
    def __init__(self, TimestepsPerEpoch, NoiseRange):
        self.TimestepsPerEpoch = TimestepsPerEpoch
        self.NoiseRange = NoiseRange

    def Run(self):
        Total = []
        # run x amount of times
        for _ in range(self.TimestepsPerEpoch):
            NewValues = self.CalcNewValues()
            #Save the results
            for i in NewValues:
                Total[i] = Total[i] + NewValues[i]

        #Take the avg
        Result = []
        for i in Total:
            Result.append(Total[i]/self.TimestepsPerEpoch)
        return Result
    
    def CalcNewValues(self):
        return [0,0,0]

    def GenerateNoise(self):
        seed()
        Noise = randint(self.NoiseRange[0], self.NoiseRange[1])
        return Noise