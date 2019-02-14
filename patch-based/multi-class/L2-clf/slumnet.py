import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy
from tflearn.metrics import Top_k 
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.normalization import batch_normalization
import tensorflow as tf
import tflearn as tfl

tf.reset_default_graph()
tflearn.init_graph(seed=100)
tf.set_random_seed(100)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_flip_updown()
img_aug.add_random_rotation(max_angle=90.)
#img_aug.add_random_crop((8, 8))
acc = Accuracy()

network = input_data(shape=[None, 40, 40, 18],data_augmentation=img_aug)
# Conv layers
network = conv_2d(network, 64, 7, strides=1, activation='relu', regularizer='L2', name = 'conv1_3_3_1', weights_init='Xavier')
network = conv_2d(network, 64, 7, strides=1, activation='relu', regularizer='L2',name = 'conv1_3_3_2', weights_init='Xavier')
#network =  batch_normalization(network)
network = conv_2d(network, 64, 7, strides=1, activation='relu', regularizer='L2', name = 'conv1_3_3_3', weights_init='Xavier')
network = dropout(network, 0.8)
network = conv_2d(network, 64, 7, strides=1, activation='relu', regularizer='L2', name = 'conv1_3_3_4', weights_init='Xavier')
network = conv_2d(network, 64, 7, strides=1, activation='relu', regularizer='L2', name = 'conv1_3_3_5', weights_init='Xavier')
#network = batch_normalization(network)
network = conv_2d(network, 64, 7, strides=1, activation='relu', regularizer='L2', name = 'conv1_3_3_6', weights_init='Xavier')
network = max_pool_2d(network, 2, strides=2)
network = conv_2d(network, 64, 7, strides=1, activation='relu', regularizer='L2', name = 'conv1_3_3_7', weights_init='Xavier')
network = max_pool_2d(network, 2, strides=2)
# Fully Connected Layer 
network = fully_connected(network, 1024, activation='relu', name = 'fc1')
# Dropout layer
network = dropout(network, 0.8)
# Fully Connected Layer 
network = fully_connected(network, 512, activation='relu', name = 'fc2')
# Dropout layer
#network = dropout(network, 1)
# Fully Connected Layer
network = fully_connected(network, 3, activation='softmax')
# Final network


network = regression(network, optimizer='adam',
                     loss="categorical_crossentropy",
                     learning_rate=0.000001, metric=acc)

# The model with details on where to save
# Will save in current directory
model = tflearn.DNN(network, checkpoint_path='../models/model-3-cl-final/model-', max_checkpoints = 3, tensorboard_verbose=1)

