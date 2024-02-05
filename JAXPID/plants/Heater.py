from plants.Plant import Plant

class Heater(Plant):
    def __init__(self, TimestepsPerEpoch, NoiseRange, StartTemp, TempOutside, TargetTemp):
        # Take general params from super
        super().__init__(TimestepsPerEpoch, NoiseRange)
        # Add these extra params
        self.StartTemp = StartTemp
        self.TempOutside = TempOutside
        self.TargetTemp = TargetTemp
        self.Temp = StartTemp

    # Take all methods from super
    # Override only this method, as this is the one that changes per plant
    def CalcNewValues(self, ControllerInput):
        # Get the temp influence form outside (k thermal transfer variable 0.12)
        TOutside = 0.12 * (self.Temp - self.TempOutside)
        # Get the temp influence from heater (M = 3.6 bc Mass of air in room)
        THeater = ControllerInput / 3.6
        # Calc the new temp with noise
        TNew = self.Temp - TOutside + THeater + super().GenerateNoise()
        # get error rate
        ErrorRate = abs(TNew - self.TargetTemp)
        
        return ErrorRate, TNew

    def Update(self, UpdateErrorRate, UpdateTemp):
        self.Temp = UpdateTemp
        super().UpdateErrorVars(UpdateErrorRate)
        return
    
    def Reset(self):
        self.Temp = self.StartTemp