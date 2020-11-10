import numpy as np
from abc import ABC,abstractmethod
from functools import reduce

class Node(ABC):
    @abstractmethod
    def activate(self,u):
        pass

    @abstractmethod
    def gradient(self,u):
        pass

class Linear(Node):
    def activate(self,u):
        return u

    def gradient(self,u):
        return 1


class Layer:
    def __init__(self,totalNodes,Node,weightInit=np.random.rand):
        self.Node       = Node()
        self.nodes      = []
        self.totalNodes = totalNodes
        self.prevLayer  = None
        self.nextLayer  = None
        self.weights    = None
        self.output     = None
        self.input      = None
        self.numInputs  = 0
        self.numOutputs = totalNodes
        self.weightInit = weightInit

    def __call__(self,u):
        return self.output(u)

    def setup(self):
        inputsPerNode = outputsPerNode = self.totalNodes
        if self.prevLayer is None:
            self.numInputs = self.prevLayer.totalNodes
        self.weights = self.weightInit(self.numOutputs,self.numInputs)

    def output(self,u):
        self.input  = u
        self.output = self.Node.activate(np.dot(self.weights,u))
        return self.output

class Network:
    def __init__(self):
        self.layers = []

    def __call__(self,u):
        y = lambda x,f:f(x)
        return reduce(y,layers,u)

    def AddLayer(self,numNodes,nodeType):
        self.layers.append(Layer(numNodes,nodeType))

    def setup(self):
        for l in self.layers:
            l.setup()


    

        
