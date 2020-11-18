import numpy as np
from ReadDataset import GetTrainingData
from NN import *

data,label = GetTrainingData()

miniD = data[:,0:5]
miniL = label[:,0:5]

numInputs = data.shape[0]

nn = Network()
nn.AddLayer(numInputs,Linear)
nn.AddLayer(15,Sigmoid)
nn.AddLayer(10,Sigmoid)
nn.setup()

Train(nn,MeanSquaredError,SquaredErrorGrad,data,label,0.001)
