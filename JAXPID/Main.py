import CONSYS

sytem = CONSYS.CONSYS( PlantNr = 1, #1 for Bathtubs, 2 for price, 3 for Aquarium
        UseNN = 1,
        NNLayers = 4, # 0 to 5 hidden layers
        NodesPerLayer = 4,
        ActivationF = 3, #1 for sigmoid, 2 for tanh, 3 for relu
        InitialValuesRange = [0.1, 0.3],
        NumberOfEpochs = 100,
        TimestepsPerEpoch = 10,
        LearningRate = 0.1,
        NoiseRange = [-0.01, 0.01],
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