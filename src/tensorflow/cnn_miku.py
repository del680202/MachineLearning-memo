

import tensorflow as tf
import numpy as np
import os
import scipy.ndimage



SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))

num = 20
imgs = []
for i in range(1, num + 1):
    imgs.append(scipy.ndimage.imread("%s/miku/%s.jpg" % (SCRIPT_PATH, i), flatten=False, mode="RGB"))
for i in range(1, num + 1):
    imgs.append(scipy.ndimage.imread("%s/miku/%s.jpg" % (SCRIPT_PATH, i), flatten=False, mode="RGB"))
imgs =  np.array(imgs)
#y_data = np.matrix(np.r_[np.ones(num), np.zeros(num)]).T
y_data = np.r_[np.c_[np.ones(num), np.zeros(num)],np.c_[np.zeros(num), np.ones(num)]]
print imgs.shape

x_test = []
for i in range(1, 11):
    x_test.append(scipy.ndimage.imread("%s/test-set/%s.jpg" % (SCRIPT_PATH, i), flatten=False, mode="RGB"))
x_test =  np.array(x_test)
#y_test = np.matrix(np.r_[np.ones(5), np.zeros(5)]).T
y_test = np.r_[np.c_[np.ones(5), np.zeros(5)],np.c_[np.zeros(5), np.ones(5)]]


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def compute_accuracy(sess, v_xs, v_ys):
    global z
    prediction = tf.nn.softmax(z)
    y_pre = sess.run(prediction, feed_dict={x: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x: v_xs, y: v_ys})
    return result


x = tf.placeholder(tf.float32, [None, 100, 100, 3]) # 400x400x3
y = tf.placeholder(tf.float32, [None, 2]) # yes or no
keep_prob_f = tf.placeholder(tf.float32)

## conv1 layer ##
W_conv1 = weight_variable([5, 5, 3, 32]) #patch 5x5, in channel size 3, out size 32
## pool1 layer ##
b_conv1 = bias_variable([32])
#Combine
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1) #output size 100x100x32
h_pool1 = max_pool_2x2(h_conv1) #output size 50x50x32
## dropout1 layer ##
#

## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64]) #patch 5x5, in channel size 32, out size 64
## pool2 layer ##
b_conv2 = bias_variable([64])
#Combine
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #output size 50x50x32
h_pool2 = max_pool_2x2(h_conv2) #output size 25x25x32
## dropout2 layer ##
#

## conv3 layer ##
W_conv3 = weight_variable([5, 5, 64, 128]) #patch 5x5, in channel size 64, out size 128
## pool2 layer ##
b_conv3 = bias_variable([128])
#Combine
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3) #output size 25x25x128
h_pool3 = max_pool_2x2(h_conv3) #output size 13x13x128
## dropout3 layer ##
#


## func1 layer ##
W_fc1 = weight_variable([13*13*128, 128])
b_fc1 = bias_variable([128])
h_pool2_flat = tf.reshape(h_pool3, [-1, 13*13*128]) #[n_samples, 13,13,128]  => [n_samples, 13*13*128]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
## dropout4 layer ##
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

### func2 layer ##
W_fc2 = weight_variable([128, 128])
b_fc2 = bias_variable([128])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
## dropout5 layer ##

#output layer
W_fc3 = weight_variable([128, 2])
b_fc3 = bias_variable([2])
z = tf.matmul(h_fc2, W_fc3) + b_fc3

#Traing
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(z, y))
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for step in range(1001):
        #print sess.run(tf.shape(h_pool3), feed_dict={x:imgs, y:y_data})
        sess.run(train_step, feed_dict={x:imgs, y:y_data})
        if step % 100 == 0:
            print "ce=",sess.run(cross_entropy, feed_dict={x:imgs, y:y_data})
            print "acc1=", compute_accuracy(sess, imgs, y_data)
            print "acc2=", compute_accuracy(sess, x_test, y_test)
