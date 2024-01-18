from JAXPID.CONSYS import CONSYS


class Main:
    def __init__(
            Plants = 1, #1 for Bathtubs, 2 for price, 3 for other
            UseNN = False,
            NNLayers = 0, # 0 to 5
            ActivationF = 1, #1 for sigmoid, 2 for tanh, 3 for relu
            InitialValuesRange = [0,1],
            NumberOfEpochs = 100,
            TimestepsPerEpoch = 1,
            LearningRate = 0.1,
            NoiseRange = [0,1],
            #Bathtub
            CrossSectionTub = 0,
            CrossSectionDrain = 0,
            HeightOfWater = 0,
            #Cournot
            MaxPrice = 100,
            ProdCost = 1,
            #Other
            var1 = 0,
            var2 = 0
    ):
        system = CONSYS()