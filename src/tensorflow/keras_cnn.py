import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

batch_size = 128
nb_classes = 10
nb_epoch   = 20
nb_data    = 28*28
log_filepath = '/tmp/keras_log'

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])

# rescale
X_train = X_train.astype(np.float32)
X_train /= 255

X_test = X_test.astype(np.float32)
X_test /= 255

# convert class vectors to binary class matrices (one hot vectors)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

old_session = KTF.get_session()

with tf.Graph().as_default():
    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)
    # build model
    model = Sequential()
    model.add(Dense(512, input_shape=(nb_data,), init='normal',name='dense1'))
    model.add(Activation('relu', name='relu1'))
    model.add(Dropout(0.2, name='dropout1'))
    model.add(Dense(512, init='normal', name='dense2'))
    model.add(Activation('relu', name='relu2'))
    model.add(Dropout(0.2, name='dropout2'))
    model.add(Dense(10, init='normal', name='dense3'))
    model.add(Activation('softmax', name='softmax1'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])

    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=1)
    cbks = [tb_cb]

    history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch = nb_epoch, verbose=1, callbacks=cbks)

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy;', score[1])


KTF.set_session(old_session)
