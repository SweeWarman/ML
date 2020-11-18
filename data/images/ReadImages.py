import numpy as np
from array import array
from matplotlib import pyplot as plt

def GetTrainingData():
  fp1 = open('train-images-idx3-ubyte','rb')   
  fp2 = open('train-labels-idx1-ubyte','rb')   

  magic1 = fp1.read(4)
  numImages = int.from_bytes(fp1.read(4),'big',signed=True)
  numRows = int.from_bytes(fp1.read(4),'big',signed=True)
  numColumns = int.from_bytes(fp1.read(4),'big',signed=True)
  pixelsPerImage = numRows*numColumns
  data = np.array(array('B',fp1.read())).rehsape(numImages,pixelsPerImage)

  magic2 = fp2.read(4)
  numImages = fp2.read(4)
  label = np.array(array('B',fp2.read())).reshape(numImages,1)
  return (data,label)













   
 
    
    

    
