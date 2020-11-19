import numpy as np
from ReadDataset import GetTrainingData
from NN import *

import BaseNN as Ref

data,label = GetTrainingData()

miniD = data[:,0:5]
miniL = label[:,0:5]

numInputs = data.shape[0]

nn = Network()
nn.AddLayer(numInputs,Linear)
nn.AddLayer(15,Sigmoid)
nn.AddLayer(10,Sigmoid)
nn.setup()

nnRef = Ref.Network([numInputs,15,10])

nnRef.weights[0] = nn.layers[1].weights[:,:-1]
nnRef.biases[0][:,0]  = nn.layers[1].weights[:,-1]
nnRef.weights[1] = nn.layers[2].weights[:,:-1]
nnRef.biases[1][:,0] = nn.layers[2].weights[:,-1]

ui = miniD[:,0].reshape(numInputs,1)

myy = nn(ui)
refy = nnRef.feedforward(ui)

yi = miniL[:,0].reshape(10,1)
dJdy = SquaredErrorGrad(myy,yi)
nn.SetCostGradient(dJdy)
nn.backwardpass(myy)


db,dw = nnRef.backprop(ui,yi)
print("error=",np.linalg.norm(myy - refy))
#Train(nn,MeanSquaredError,SquaredErrorGrad,data,label,0.001)
