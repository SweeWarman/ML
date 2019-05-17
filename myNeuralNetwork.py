import numpy as np


class Node:
    def __init__(self, rank, W, numOuts):
        self.rank = rank
        self.W = W
        self.U = None
        self.numOuts = numOuts
        self.gradientsW = None
        self.du_dw = None
        self.dJ_du = 0
        self.activation = 0

    def sigmoidActivation(self, x):
        return 1/(1 + np.exp(-1*x))

    def computeActivation(self, U):
        self.U = U
        u = np.dot(self.W, U)
        y = self.sigmoidActivation(u)
        self.activation = y
        return y

    def computeGradient(self, nextlayer, nextdJ_du=0):

        y = self.activation
        dy_du = y*(y-1)

        self.du_dw = (dy_du * self.U)

        if nextlayer is not None:
            for node in nextlayer.Nodes:
                dyn_dy = node.activation * node.W[self.rank]
                dJ_dy = node.dJ_du
                self.dJ_du += dJ_dy*dyn_dy
        else:
            self.dJ_du = nextdJ_du

        self.gradientsW = self.dJ_du * self.du_dw


class Layer:
    def __init__(self, numNodesPerLayer, numInputs, inputLayer=False,
                 outputLayer=False):
        self.numInputs = numInputs
        self.numNodesPerLayer = numNodesPerLayer
        self.Nodes = []  # type: [Node]
        self.nextlayer = None
        self.inputLayer = inputLayer
        self.outputLayer = outputLayer
        self.activations = []
        for i in range(self.numNodesPerLayer):
            randW = []
            if self.inputLayer:
                self.Nodes = []
            elif self.nextlayer is not None:
                self.Nodes.append(Node(i, randW, self.nextlayer.numInputs))
            else:
                self.Nodes.append(Node(i, randW, self.numOutputs))

    def GetActivations(self, U):
        if not self.inputLayer:
            for node in self.Nodes:
                self.activations.append(node.ComputeActivation(U))
        else:
            self.activations = U
        return self.activations

    def GetGradients(self, nextlayer, GradOutput):

        for i, node in enumerate(self.Nodes):
            if self.outputLayer:
                node.computeGradient(None, nextdJ_du=GradOutput[i])
            else:
                node.computeGradient(nextlayer)

        return 0


class Network:
    def __init__(self, numLayers, numNodesPerLayer):
        """
        @param numLayers: number of layers in the network
        @param numNodesPerLayer: list indicating nodes per layer

        """
        self.layers = []  # type: [Layer]
        self.output = []

        for i, numNodes in enumerate(numNodesPerLayer):
            if i == 0:
                numInputs = numNodesPerLayer
                self.layers.append(Layer(numNodes, numInputs, inputLayer=True))
            elif i == (len(self.layers) - 1):
                numInputs = self.layers[i-1].numNodesPerLayer
                self.layers.append(Layer(numNodes, numInputs, outputLayer=True))
            else:
                numInputs = self.layers[i-1].numNodesPerLayer
                self.layers.append(Layer(numNodes, numInputs))

    def ForwardPass(self, U):

        prevActivations = U
        for layer in self.layers:
            prevActivations = layer.GetActivations(prevActivations)

        self.output = prevActivations

        return self.output

    def BackwardPass(self):
        return 0
