import numpy as np
from tflearn.data_utils import shuffle
import slumnet as net

# Read files 
trainX = np.load('/home/kgadira/data/filtered-3p-tr-X')
trainY = np.load('/home/kgadira/data/filtered-3p-tr-Y')
testX = np.load('/home/kgadira/data/filtered-3p-te-X')
testY = np.load('/home/kgadira/data/filtered-3p-te-Y')

print trainX.shape, trainY.shape, testX.shape, testY.shape

print trainY, testY

# One - Hot encode trainY
for i in range(trainY.shape[0]):
    curr = np.zeros((1,2))
    curr[0,trainY[i]] = 1
    if i == 0:
       final_train_Y = curr
    else:
       final_train_Y = np.vstack((final_train_Y, curr))
print final_train_Y.shape

trainY = final_train_Y


model = net.model

trainX = trainX.reshape([-1, 40, 40, 18])
print trainX.shape

# Shuffle data
trainX, trainY = shuffle(trainX, trainY)

# Train the model
model.fit(trainX, trainY, n_epoch=256,validation_set= 0.2, show_metric=True, run_id="deep_nn1", batch_size = 128,snapshot_step=100000)
model.save("../../models/filtered-model-3-cl-final/final-model.tflearn")
