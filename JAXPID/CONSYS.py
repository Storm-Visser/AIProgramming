from plants.BathtubPlant import BathtubPlant
from plants.CournotPlant import CournotPlant
from plants.OtherPlant import OtherPlant
from controllers.Controller import Controller
from controllers.NNController import NNController
import matplotlib.pyplot as plt


class CONSYS:
    def __init__(self, PlantNr, UseNN, NNLayers, NodesPerLayer, ActivationF, InitialValuesRange, NumberOfEpochs, TimestepsPerEpoch, LearningRate, 
                 NoiseRange, CrossSectionTub, CrossSectionDrain, HeightOfWater, MaxPrice, ProdCost,  var1, var2):
        #General
        self.PlantNr = PlantNr
        self.UseNN = UseNN
        self.NNLayers = NNLayers
        self.NodesPerLayer = NodesPerLayer
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
        self.KValues = []

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
            self.Controller = NNController(self.LearningRate, self.NNLayers, self.NodesPerLayer, self.ActivationF, self.InitialValuesRange)
        else:
            self.Controller = Controller(self.LearningRate)

        self.RunSim()

    def RunSim(self):
        ControllerInput = [0,0,0] #updates every timestep, start at 0
        for _ in range(self.NumberOfEpochs):
            PlantEr = 0
            PlantUpd = 0
            PlantErTot = 0
            NewKValues = [0,0,0]
            for _ in range(self.TimestepsPerEpoch):
                # get results from epoch 
                PlantEr, PlantUpd = self.Plant.Run(ControllerInput[0], ControllerInput[1], ControllerInput[2])
                PlantErTot = PlantErTot + (PlantEr**2)
                #update the plant values
                self.Plant.Update(PlantEr, PlantUpd)                
                # use results to calc new input
                ControllerInput = self.Controller.Analyse(self.Plant.Run)
                NewKValues = ControllerInput
            
            #reset
            self.Plant.Reset()
            #save results
            self.ErrorRate.append(PlantErTot/self.TimestepsPerEpoch)
            self.KValues.append(NewKValues)

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

        # Create a single graph for all points
        # Extract the values for each column (Array 1, Array 2, Array 3)
        array1_values = [sublist[0] for sublist in self.KValues]
        array2_values = [sublist[1] for sublist in self.KValues]
        array3_values = [sublist[2] for sublist in self.KValues]

        ax2.plot(time_steps, array1_values, label='K1')
        ax2.plot(time_steps, array2_values, label='K2')
        ax2.plot(time_steps, array3_values, label='K3')

        ax2.set_title('K Values')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Values')
        ax2.legend()

        plt.tight_layout()
        plt.show()

        
        

        
