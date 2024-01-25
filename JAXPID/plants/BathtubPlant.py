from plants.Plant import Plant
import jax
import jax.numpy as jnp
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
    # Returns Error rate
    def CalcNewValues(self, ControllerInput):
        # Removed
        VWater = jnp.sqrt(2 * self.g * self.HeightOfWater)
        FlowRate = VWater * self.CrossSectionDrain
        # Added
        Faucet = super().GenerateNoise() + ControllerInput
        # Result
        NewVolume = Faucet - FlowRate
        # Account for tub area
        NewHeightOfWater = self.HeightOfWater + NewVolume/self.CrossSectionTub
        # Feedback
        ErrorRate = abs(((NewHeightOfWater - self.TargetHeight) / self.TargetHeight))
        return ErrorRate, NewHeightOfWater

    def Update(self, UpdateErrorRate, UpdateWaterHeigt):
        self.HeightOfWater = UpdateWaterHeigt
        super().UpdateErrorVars(UpdateErrorRate)
        return
    
    def Reset(self):
        self.HeightOfWater = self.TargetHeight