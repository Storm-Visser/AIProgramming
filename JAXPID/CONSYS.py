from plants.BathtubPlant import BathtubPlant
from plants.CournotPlant import CournotPlant
from plants.OtherPlant import OtherPlant
from Controllers.Controller import Controller
from Controllers.NNController import NNController
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
        self.K1 = []
        self.K2 = []
        self.K3 = []

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
        ControllerInput = [0,0,0] #updates every timestep, start at 0
        for _ in range(self.NumberOfEpochs):
            PlantEr = 0
            PlantUpd = 0
            PlantErTot = 0
            for _ in range(self.TimestepsPerEpoch):
                # get results from epoch 
                PlantEr, PlantUpd = self.Plant.Run(ControllerInput[0], ControllerInput[1], ControllerInput[2])
                PlantErTot = PlantErTot + (PlantEr**2)
                #update the plant values
                self.Plant.Update(PlantEr, PlantUpd)                
                # use results to calc new input
                ControllerInput = self.Controller.Analyse(self.Plant.Run)
                self.K1.append(ControllerInput[0])
                self.K2.append(ControllerInput[1])
                self.K3.append(ControllerInput[2])
            #reset
            self.Plant.Reset()
            #save results
            self.ErrorRate.append(PlantErTot/self.TimestepsPerEpoch)

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

        #Plot results
        ax2.plot(time_steps, self.K1, label='K1')
        ax2.plot(time_steps, self.K2, label='K2')
        ax2.plot(time_steps, self.K3, label='K3')
        ax2.set_title('Results Over Time')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Results')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()  # Adjusts spacing between subplots
        plt.show()

        
        

        
