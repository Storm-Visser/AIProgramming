import plants
import Controllers

class CONSYS:
    def __init__(self, PlantNr, UseNN, NNLayers, ActivationF, InitialValuesRange, NumberOfEpochs, TimestepsPerEpoch, LearningRate, 
                 NoiseRange, CrossSectionTub, CrossSectionDrain, HeightOfWater, MaxPrice, ProdCost,  var1, var2):
        #General
        self.PlantNr = PlantNr
        self.UseNN = UseNN
        self.NNLayers = NNLayers
        self.ActivationF = ActivationF
        self.InitialValuesRange = InitialValuesRange
        self.NumberOfEpochs = NumberOfEpochs
        self.TimestepsPerEpoch = TimestepsPerEpoch
        self.LearningRate = LearningRate
        self.NoiseRange = NoiseRange
        #Bathtub
        self.CrossSectionTub = CrossSectionTub
        self.CrossSectionDrain = CrossSectionDrain
        self.HeightOfWater = HeightOfWater 
        #Cournot
        self.MaxPrice = MaxPrice
        self.ProdCost = ProdCost
        #Other
        self.var1 = var1 
        self.var2 = var2      
        #Results
        self.ErrorRate = []
        self.Results = [] #2dArray array with timestepNr[param1,param2,param3(,param4)]

        self.StartSim()

    def StartSim(self):
        # set Plant
        if (self.PlantNr == 1):
            self.Plant = plants.BathtubPlant(self.TimestepsPerEpoch, self.NoiseRange, self.CrossSectionTub, self.CrossSectionDrain, self.HeightOfWater)
        elif (self.PlantNr == 2):
            self.Plant = plants.PricePlant(self.TimestepsPerEpoch, self.NoiseRange, self.MaxPrice, self.ProdCost)
        elif (self.PlantNr == 3):
            self.Plant = plants.Otherplant(self.TimestepsPerEpoch, self.NoiseRange, self.var1, self.var2)

        # set Controller
        if self.useNN:
            self.Controller = Controllers.NNController(self.LearningRate, self.NNLayers, self.ActivationF, self.InitialValuesRange)
        else:
            self.Controller = Controllers.PIDController(self.LearningRate)

        self.RunSim()

    def RunSim(self):
        ControllerInput = 0 #updates every timestep
        for _ in range(self.NumberOfEpochs):
            # get results from epoch 
            PlantResults = self.Plant.Run(ControllerInput)

            #save results for visualisation
            self.ErrorRate.append(PlantResults[1])
            self.Results.append([PlantResults[2], PlantResults[3], ControllerInput])

            # use results to calc new input
            ControllerInput =  self.Controller.Analyze(PlantResults)

        self.ShowResults()
    
    def ShowGraphs(self):
        print(self.ErrorRate)
        print(self.Results)
        

        
