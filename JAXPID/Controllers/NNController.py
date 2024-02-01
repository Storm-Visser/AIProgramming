from controllers.Controller import Controller
from controllers.Node import Node
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad, vmap

class NNController(Controller):
    def __init__(self, LearningRate, NNLayers, NodesPerLayer, ActivationF, InitialValuesRange):
        super().__init__(LearningRate)
        self.NNLayers = NNLayers
        self.NodesPerLayer = NodesPerLayer
        self.ActivationF = ActivationF #todo add this to the nodes
        self.InitialValuesRange = InitialValuesRange

        self.HiddenLayers = []
        self.InputNodes = []
        self.OutputNode = None
        self.Weights = None
        self.Biases = None
        self.SetupNN()

        

    def Analyse(self, ErrorRate):
        # set new input nodes from Errorrate
        # Do a forward pass and get the value for the plant
        FwPassOutput = self.Forward(self.InputValue)

        # Do a backward pass and get the gradient values
        _, Gradients = self.Backward(self.InputValue, ErrorRate)

        # Update the weights using the gradients 
        self.UpdateWeights(Gradients, self.LearningRate)

        # Return the new input for the next timestep of the plant
        return FwPassOutput

    def SetupNN(self):
        # setup the starting nodes
        for _ in range(3):
            NewNode = Node()
            NewNode.AddOutput(1)
            self.InputNodes.append(NewNode)

        self.OutputNode = Node()

        # setup the layers and the nodes
        for _ in range(self.NNLayers):
                Layer = [Node() for _ in range(self.NNLayers)]
                self.HiddenLayers.append(Layer)

        # Connect input nodes to the first hidden layer
        for i, Node in enumerate(self.InputNodes):
            for HiddenNode in self.HiddenLayers[0]:
                Weight = np.random.randn()
                HiddenNode.AddInput(Node, Weight)

        # Connect hidden layers
        for i in range(len(self.HiddenLayers) - 1):
            for Node1 in self.HiddenLayers[i]:
                for Node2 in self.HiddenLayers[i + 1]:
                    weight = np.random.randn()
                    Node2.AddInput(Node1, weight)

        # Connect last hidden layer to the output node
        for Node in self.HiddenLayers[-1]:
            Weight = np.random.randn()
            self.OutputNode.AddInput(Node, Weight)

        # Initialize weights and biases as JAX arrays with the same structure as gradients
        self.Weights = [[jnp.array(Node.Weights) for Node in Layer] for Layer in self.HiddenLayers] + [jnp.array(self.OutputNode.Weights)]
        self.Biases = [[jnp.array(Node.Bias) for Node in Layer] for Layer in self.HiddenLayers] + [jnp.array([self.OutputNode.Bias])]

    def updateInputData(self, ErrorRate, PrevErrorRate, ErrorRateSum):
        self.InputValue = [ErrorRate, PrevErrorRate, ErrorRateSum]
        

    def Forward(self, InputData):
        for i, Node in enumerate(self.InputNodes):
            Node.output = InputData[i]

        for HiddenLayer, Bias in zip(self.HiddenLayers, self.Biases):
            for Node, NodeBias in zip(HiddenLayer, Bias):
                Node.bias = NodeBias
                Node.CalcOutput()

        self.OutputNode.CalcOutput()
        return self.OutputNode.output

    def Backward(self, InputData, Target):
        def LossFn(Params):
            # forward pass
            Output = self.Forward(InputData)
            # get the loss
            Loss = self.Loss(Target, Output)
            return Loss
        # calculate the gradients with jax
        Grads = jax.grad(LossFn)(self.GetParams())
        return LossFn(self.GetParams()), Grads

    def Loss(self, Target, Output):
        return jnp.mean((Output - Target) ** 2)

    def GetParams(self):
        return [Node.Weights for Layer in self.HiddenLayers for Node in Layer] + [self.OutputNode.Weights]

    def UpdateWeights(self, Gradients, LearningRate):
        # Iterate over the layers and nodes to update weights using gradients
        for i in range(len(self.HiddenLayers)):
            for j in range(len(self.HiddenLayers[i])):
                self.Weights[i][j] -= LearningRate * Gradients[i][j]

        # Update the weights of the OutputNode
        print(Gradients[-1][1])
        self.OutputNode.Weights -= LearningRate * Gradients[-1]


        

