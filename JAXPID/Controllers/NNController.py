import Controller

class NNController(Controller):
    def __init__(self, LearningRate, NNLayers, ActivationF, InitialValuesRange):
        super.__init__(LearningRate)
        self.NNLayers = NNLayers
        self.ActivationF = ActivationF
        self.InitialValuesRange = InitialValuesRange
        

