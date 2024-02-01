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
        self.ActivationF = ActivationF #add this to the nodes
        self.InitialValuesRange = InitialValuesRange
        #NN stuff
        self.HiddenLayers = []
        self.InputNodes = []
        self.OutputNode = None

        self.SetupNN()
        

    def Analyse(self, ErrorRate):
        # set new input nodes from Errorrate
        # Do a forward pass and get the value for the plant
        FwPassOutput = self.forward(self.InputValue)

        # Do a backward pass and get the gradient values
        gradients = self.backward(self.InputValue, ErrorRate)

        # Update the weights using the gradients 
        print(gradients)
        self.update_weights(gradients, self.LearningRate)

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

        # Initialize weights and biases as JAX arrays
        self.weights = [jnp.array(node.weights) for layer in self.HiddenLayers for node in layer] + [jnp.array(self.OutputNode.weights)]
        self.biases = [jnp.array([node.bias for node in layer]) for layer in self.HiddenLayers] + [jnp.array([self.OutputNode.bias])]

    def updateInputData(self, errorRate, PrevErrorRate, ErrorRateSum):
        self.InputValue = [errorRate, PrevErrorRate, ErrorRateSum]
        

    def forward(self, input_data):
        for i, node in enumerate(self.InputNodes):
            node.output = input_data[i]

        for HiddenLayer, bias in zip(self.HiddenLayers, self.biases):
            for node, nodeBias in zip(HiddenLayer, bias):
                node.bias = nodeBias
                node.CalcOutput()

        self.OutputNode.CalcOutput()
        return self.OutputNode.output

    def backward(self, inputData, target):
        def loss_fn(params):
            # Forward pass through the network
            output = self.forward(inputData)
            # Calculate the loss
            loss = self.loss(target, output)
            return loss

        # Compute the gradient of the loss with respect to the neural network parameters
        grads = jax.grad(loss_fn)(self.get_params())

        return loss_fn(self.get_params()), grads

    def loss(self, target, output):
        return jnp.mean((output - target) ** 2)

    def get_params(self):
        return [node.weights for layer in self.HiddenLayers for node in layer] + [self.OutputNode.weights]

    def update_weights(self, gradients, learning_rate):
        # Iterate over the layers
        for i, (layer_gradients, layer_weights) in enumerate(zip(gradients, self.weights)):
            # Iterate over nodes within the layer
            for j, (node_gradients, node_weights) in enumerate(zip(layer_gradients, layer_weights)):
                # Update node weights using gradients
                node_weights -= learning_rate * node_gradients
                # Assign the updated weights back to the network weights
                self.weights[i][j] = node_weights

        # Update the weights of the OutputNode
        self.OutputNode.weights -= learning_rate * gradients[-1]

        # Reshape the updated weights of the OutputNode
        self.OutputNode.weights = self.OutputNode.weights.reshape(self.OutputNode.weights.shape)


        

