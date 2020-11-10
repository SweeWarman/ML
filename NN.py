import numpy as np
from abc import ABC,abstractmethod
from functools import reduce

class Node(ABC):
    @abstractmethod
    def activate(self,u):
        pass

    @abstractmethod
    def gradient(self,x,y):
        pass

class Linear(Node):
    def activate(self,u):
        return u

    def gradient(self,x,y):
        return 1

class Sigmoid(Node):
    def activate(self,u):
        return 1/(1+np.exp(u))

    def gradient(self,x,y):
        return y*(1-y)

class Layer:
    def __init__(self,totalNodes,Node,weightInit=np.random.rand):
        self.Node       = Node()
        self.nodes      = []
        self.totalNodes = totalNodes
        self.prevLayer  = None
        self.nextLayer  = None
        self.weights    = None
        self.outputs    = None
        self.input      = None
        self.numInputs  = 0
        self.numOutputs = totalNodes
        self.weightInit = weightInit

    def __call__(self,u):
        return self.output(u)

    def setup(self):
        if self.prevLayer is not None:
            self.numInputs = self.prevLayer.totalNodes
            self.weights = self.weightInit(self.numOutputs,self.numInputs)

    def output(self,u):
        self.input  = u
        if self.weights is None:
            self.outputs = u
            return self.outputs
        self.outputs = self.Node.activate(np.dot(self.weights,u))
        return self.outputs

class Network:
    def __init__(self):
        self.layers = []

    def __call__(self,u):
        y = lambda x,f:f(x)
        return reduce(y,self.layers,u)

    def AddLayer(self,numNodes,nodeType):
        newLayer = Layer(numNodes,nodeType)
        if len(self.layers) > 0:
            self.layers[-1].nextLayer = newLayer
            newLayer.prevLayer = self.layers[-1]
        self.layers.append(newLayer)

    def setup(self):
        for l in self.layers:
            l.setup()


    

        
