import numpy as np
from abc import ABC,abstractmethod

class Node(ABC):
    def __init__(self,nInputs,nOutputs):
        self.numInputs    = nInputs + 1
        self.numOutputs   = nOutputs
        self.gradients    = np.zeros((nOutputs,nInputs))
        self.weights      = np.zeros((1,numInputs))
        self.outputs      = np.zeros((nOutputs,1))
        self.inputs       = np.zeros((nInputs,1))
    
    @abstractmethod
    def activate(self,u):
        pass

    @abstractmethod
    def gradient(self,u):
        pass

class Linear(Node):
    def __init__(self,nInputs,nOutputs):
        super().__init__(nInputs,nOutputs)
        self.weights = np.ones((self.numInputs,1))

    def activate(self,u):
        return u

    def gradient(self,u):
        return np.ones((self.numOutputs,self.numInputs))


class Layer:
    def __init__(self,totalNodes,Node):
        self.Node       = Node
        self.nodes      = []
        self.totalNodes = totalNodes
        self.prevLayer  = None
        self.nextLayer  = None
        self.weights    = None

    def setup(self):
        inputsPerNode = outputsPerNode = self.totalNodes
        if self.prevLayer is None:
            inputsPerNode = self.prevLayer.totalNodes
        if self.nextLayer is None:
            outputsPerNode = self.nextLayer.totalNodes
        self.nodes = [Node(inputsPerNode,outputsPerNode) for i in range(totalNodes)]
        for i,node in enumerate(self.nodes):
            if i > 0:
                self.weights = np.vstack((self.weights,node.weights))
            else:
                self.weights = node.weights

    def output(self,u):
        return self.Node.activate(np.dot(self.weights,u))

class Network:
    def __init__(self):
        self.layers = []
        
    def AddLayer(self,numNodes,nodeType):
        self.layers.append(Layer(numNodes,nodeType))

    def setup(self):
        for l in self.layers:
            l.setup()

    

        
