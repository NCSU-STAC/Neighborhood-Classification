import numpy as np
from tflearn.data_utils import shuffle
import buildingnet as net

# Read files 

trainX = np.load('/home/kgadira/data/final-pt-tr-2-X')
trainY = np.load('/home/kgadira/data/final-pt-tr-2-Y')
testX = np.load('/home/kgadira/data/final-pt-te-2-X')
testY = np.load('/home/kgadira/data/final-pt-te-2-Y')

print trainX.shape, trainY.shape, testX.shape, testY.shape

print np.max(trainY), np.max(testY)


model = net.model

trainX = trainX.reshape([-1, 40, 40, 18])
print trainX.shape

# Shuffle data
trainX, trainY = shuffle(trainX, trainY)

# Train the model
model.fit(trainX, trainY, n_epoch=256,validation_set= 0.2, show_metric=True, run_id="deep_nn1", batch_size = 128,snapshot_step=100000)
model.save("../models/filtered-model-2-cl-final/final-model.tflearn")

