import numpy as np
from array import array
from matplotlib import pyplot as plt

def GetTrainingData():
  fp1 = open('data/train-images-idx3-ubyte','rb')   
  fp2 = open('data/train-labels-idx1-ubyte','rb')   

  magic1 = fp1.read(4)
  numImages = int.from_bytes(fp1.read(4),'big',signed=True)
  numRows = int.from_bytes(fp1.read(4),'big',signed=True)
  numColumns = int.from_bytes(fp1.read(4),'big',signed=True)
  pixelsPerImage = numRows*numColumns
  bytearray1 = array('B',fp1.read())
  data = np.array(bytearray1).reshape(numImages,pixelsPerImage)

  magic2 = fp2.read(4)
  numImages = int.from_bytes(fp2.read(4),'big',signed=True)
  bytearray2 = array('B',fp2.read())
  label = np.array(bytearray2).reshape(numImages,1)
  extlabel = np.zeros((numImages,10))
  for i in range(numImages):
    extlabel[i,label[i]] = 1
  return (data.T,extlabel.T)













   
 
    
    

    
