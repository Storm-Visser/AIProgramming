from plants.Plant import Plant
import numpy as np

class BathtubPlant(Plant):

    g = 9.81
    TargetHeight = 0

    def __init__(self, TimestepsPerEpoch, NoiseRange, CrossSectionTub, CrossSectionDrain, HeightOfWater):
        # Take general params from super
        super().__init__(TimestepsPerEpoch, NoiseRange)
        # Add these extra params
        self.CrossSectionTub = CrossSectionTub
        self.CrossSectionDrain = CrossSectionDrain
        self.HeightOfWater = HeightOfWater
        self.TargetHeight = HeightOfWater

    # Take all methods from super
    # Override only this method, as this is the one that changes per plant
    # Returns Error rate (%), Target height, Actual height, Noise
    def CalcNewValues(self, ControllerInput):
        # Remove
        VWater = np.sqrt(2 * self.g * self.HeightOfWater)
        FlowRate = VWater * self.CrossSectionDrain
        # Add
        NewVolume = super().GenerateNoise() + ControllerInput - FlowRate
        # Account for tub area
        self.HeightOfWater = self.HeightOfWater + NewVolume/self.CrossSectionTub
        # Feedback
        ErrorRate = ((self.HeightOfWater/self.TargetHeight) * 100) - 100
        print("Er: " + str(ErrorRate) + "%")
        print("WH: " + str(self.HeightOfWater))
        print()
        return ErrorRate