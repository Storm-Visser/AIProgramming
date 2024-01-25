from plants.BathtubPlant import BathtubPlant
from plants.CournotPlant import CournotPlant
from plants.OtherPlant import OtherPlant
from controllers.Controller import Controller
from controllers.NNController import NNController
import matplotlib.pyplot as plt


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
            self.Plant = BathtubPlant(self.TimestepsPerEpoch, self.NoiseRange, self.CrossSectionTub, self.CrossSectionDrain, self.HeightOfWater)
        elif (self.PlantNr == 2):
            self.Plant = CournotPlant(self.TimestepsPerEpoch, self.NoiseRange, self.MaxPrice, self.ProdCost)
        elif (self.PlantNr == 3):
            self.Plant = OtherPlant(self.TimestepsPerEpoch, self.NoiseRange, self.var1, self.var2)

        # set Controller
        if self.UseNN:
            self.Controller = NNController(self.LearningRate, self.NNLayers, self.ActivationF, self.InitialValuesRange)
        else:
            self.Controller = Controller(self.LearningRate)

        self.RunSim()

    def RunSim(self):
        ControllerInput = 0 #updates every timestep, start at 0
        for _ in range(self.NumberOfEpochs):
            PlantEr = 0
            PlantUpd = 0
            for _ in range(self.TimestepsPerEpoch):
                # get results from epoch 
                PlantEr, PlantUpd = self.Plant.Run(ControllerInput)
                self.Plant.Update(PlantUpd)                
                # use results to calc new input
                ControllerInput, K1, K2, K3 = self.Controller.Analyse(self.Plant.Run, PlantEr)
            #save results
            self.ErrorRate.append((PlantEr)**2)
            self.Results.append(PlantUpd)

        self.ShowResults()
    
    def ShowResults(self):
        print("Done")

        time_steps = range(len(self.ErrorRate))

        # Create subplots
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        # Plot error rates
        ax1.plot(time_steps, self.ErrorRate, label='Error Rate')
        ax1.set_title('Error Rate Over Time')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Error Rate')
        ax1.legend()
        ax1.grid(True)

        # Plot results
        ax2.plot(time_steps, self.Results, label='Result')
        ax2.set_title('Results Over Time')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Results')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()  # Adjusts spacing between subplots
        plt.show()

        
        

        
