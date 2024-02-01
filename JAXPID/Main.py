import CONSYS

sytem = CONSYS.CONSYS( PlantNr = 1, #1 for Bathtubs, 2 for price, 3 for other
        UseNN = True,
        NNLayers = 2, # 0 to 5 hidden layers
        NodesPerLayer = 3,
        ActivationF = 1, #1 for sigmoid, 2 for tanh, 3 for relu
        InitialValuesRange = [0,1],
        NumberOfEpochs = 10,
        TimestepsPerEpoch = 10,
        LearningRate = 0.01,
        NoiseRange = [-0.5,0.5],
        #Bathtub
        CrossSectionTub = 10,
        CrossSectionDrain = 0.1,
        HeightOfWater = 30.0, # both starting height and target height
        #Cournot
        MaxPrice = 100,
        ProdCost = 1,
        #Other
        var1 = 0,
        var2 = 0
        )