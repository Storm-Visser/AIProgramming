from plants.Plant import Plant


class CournotPlant(Plant):
    def __init__(self, TimestepsPerEpoch, NoiseRange, MaxPrice, ProdCost, TargetProfit):
        # Take general params from super
        super().__init__(TimestepsPerEpoch, NoiseRange)
        # Add these extra params
        self.MaxPrice = MaxPrice
        self.ProdCost = ProdCost
        self.Target = TargetProfit
        self.Q1 = 0.1
        self.Q2 = 0.1

    # Take all methods from super
    # Override only this method, as this is the one that changes per plant
     # Take all methods from super
    def CalcNewValues(self, ControllerInput):
        # New Q values
        NewQ1 = super().SigmoidMap(self.Q1 + ControllerInput)
        NewQ2 = super().SigmoidMap(self.Q2 + super().GenerateNoise())
        # Sum of Qs, mapped Qs to sigmoid to keep it between 0 and 1
        QSum = NewQ1 + NewQ2
        # Get the new prices
        Price = self.MaxPrice - QSum
        # Calc profit
        Profit = NewQ1 * (Price - self.ProdCost)
        # Get error rate
        ErrorRate = abs(self.Target - Profit)
        
        return ErrorRate, [NewQ1, NewQ2]

    def Update(self, UpdateErrorRate, UpdateQs):
        self.Q1 = UpdateQs[0]
        self.Q2 = UpdateQs[1]
        super().UpdateErrorVars(UpdateErrorRate)
        return
    
    def Reset(self):
        self.Q1 = 0.0
        self.Q2 = 0.0

    