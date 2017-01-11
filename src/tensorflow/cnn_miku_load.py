
import numpy as np

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import load_image




# Building convolutional network
network = input_data(shape=[None, 100, 100, 3], name='input')
network = conv_2d(network, 64, 5, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 128, 5, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 512, activation='relu')
#network = dropout(network, 0.8)
network = fully_connected(network, 1024, activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.00001,
                     loss='categorical_crossentropy', name='target')

model = tflearn.DNN(network)
model.load('miku_model.tflearn')


imgs = []
num = 4
for i in range(1, num + 1):
    img = load_image("/tmp/t%s.jpg" % (i))
    img = img.resize((100,100))
    img_arr = np.asarray(img)
    imgs.append(img_arr)
imgs = np.array(imgs)
print imgs.shape
print np.round(model.predict(imgs))
