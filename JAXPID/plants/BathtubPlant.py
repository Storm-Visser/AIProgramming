import Plant
import numpy as np

class BathtubPlant(Plant):
    def __init__(self, TimestepsPerEpoch, NoiseRange, CrossSectionTub, CrossSectionDrain, HeightOfWater):
        # Take general params from super
        super.__init__(TimestepsPerEpoch, NoiseRange)
        # Add these extra params
        self.CrossSectionTub = CrossSectionTub
        self.CrossSectionDrain = CrossSectionDrain
        self.HeightOfWater = HeightOfWater

    # Take all methods from super
    # Override only this method, as this is the one that changes per plant
    def CalcNewValues(self):
        Noise = super.GenerateNoise()
        
        return