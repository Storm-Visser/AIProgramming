from controllers.Controller import Controller
from controllers import Node
import numpy as np

class NNController(Controller):
    def __init__(self, LearningRate, NNLayers, NodesPerLayer, ActivationF, InitialValuesRange):
        super.__init__(LearningRate)
        self.NNLayers = NNLayers
        self.NodesPerLayer = NodesPerLayer
        self.ActivationF = ActivationF #add this to the nodes
        self.InitialValuesRange = InitialValuesRange
        self.HiddenLayers = []
        self.InputNodes = []
        self.OutputNode = 0
        self.SetupNN()
        

    def Analyse(self, ErrorRate):
        NewInputValue = 0
        # nn shit
        return NewInputValue

    def SetupNN(self):
        # setup the starting nodes
        for _ in range(3):
            NewNode = Node()
            NewNode.AddOutput(1)
            self.InputNodes.append(Node())

        self.OutputNode = Node()

        # setup the layers and the nodes
        for _ in range(self.NNLayers):
                layer = [Node() for _ in range(self.NNLayers)]
                self.HiddenLayers.append(layer)

        # Connect input nodes to the first hidden layer
        for i, node in enumerate(self.InputNodes):
            for HiddenNode in self.HiddenLayers[0]:
                weight = np.random.randn()
                HiddenNode.AddInput(node, weight)

        # Connect hidden layers
        for i in range(len(self.HiddenLayers) - 1):
            for node1 in self.HiddenLayers[i]:
                for node2 in self.HiddenLayers[i + 1]:
                    weight = np.random.randn()
                    node2.AddInput(node1, weight)

        # Connect last hidden layer to the output node
        for node in self.HiddenLayers[-1]:
            weight = np.random.randn()
            self.OutputNode.AddInput(node, weight)

    def forward(self, InputData):
        for i, node in enumerate(self.InputNodes):
            node.output = InputData[i]

        for HiddenLayers in self.HiddenLayers:
            for node in HiddenLayers:
                node.CalcOutput()

        self.OutputNode.CalcOutput()
        return self.OutputNode.output

        

