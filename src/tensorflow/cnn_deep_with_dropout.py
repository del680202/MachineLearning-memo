from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)

def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
#print(x_image.shape) #[n_samples, 28, 28, 1]


## conv1 layer ##
W_conv1 = weight_variable([5, 5, 1, 32]) #patch 5x5, in channel size 1, out size 32
## pool1 layer ##
b_conv1 = bias_variable([32])
#Combine
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1) #output size 14x14x32


## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64]) #patch 5x5, in channel size 32, out size 64
## pool2 layer ##
b_conv2 = bias_variable([64])
#Combine
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #output size 14x14x32
h_pool2 = max_pool_2x2(h_conv2) #output size 7x7x32


## conv3 layer ##
W_conv3 = weight_variable([5, 5, 64, 128]) #patch 5x5, in channel size 64, out size 128
## pool2 layer ##
b_conv3 = bias_variable([128])
#Combine
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3) #output size 7x7x128
h_pool3 = max_pool_2x2(h_conv3) #output size 4x4x128

## func1 layer ##
W_fc1 = weight_variable([4*4*128, 128])
b_fc1 = bias_variable([128])
h_pool2_flat = tf.reshape(h_pool3, [-1, 4*4*128]) #[n_samples, 4,4,128]  => [n_samples, 4*4*128]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

### func2 layer ##
W_fc2 = weight_variable([128, 128])
b_fc2 = bias_variable([128])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
## dropout5 layer ##
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

#output layer
W_fc3 = weight_variable([128, 10])
b_fc3 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
sess.run(tf.initialize_all_variables())

for i in range(40001):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 1000 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
