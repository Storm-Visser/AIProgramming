from plants.Plant import Plant

class CournotPlant(Plant):
    def __init__(self, TimestepsPerEpoch, NoiseRange, MaxPrice, ProdCost):
        # Take general params from super
        super.__init__(TimestepsPerEpoch, NoiseRange)
        # Add these extra params
        self.MaxPrice = MaxPrice
        self.ProdCost = ProdCost

    # Take all methods from super
    # Override only this method, as this is the one that changes per plant
    def CalcNewValues(self):
        Noise = super().GenerateNoise()
        #add math here
        return