import CONSYS

sytem = CONSYS.CONSYS( PlantNr = 2, #1 for Bathtubs, 2 for price, 3 for Aquarium
        UseNN = 0,
        NNLayers = 1, # 0 to 5 hidden layers
        NodesPerLayer = 3,
        ActivationF = 3, #1 for sigmoid, 2 for tanh, 3 for relu
        InitialValuesRange = [0.1, 0.3],
        NumberOfEpochs = 30,
        TimestepsPerEpoch = 10,
        LearningRate = 0.01,
        NoiseRange = [-0.1, 0.1],
        #Bathtub
        CrossSectionTub = 10,
        CrossSectionDrain = 0.1,
        HeightOfWater = 30.0, # both starting height and target height
        #Cournot
        MaxPrice = 10,
        ProdCost = 0.1,
        TargetProfit = 8,
        #Other
        StartTemp = 20,
        TempOutside = -5,
        TargetTemp = 20,
        )