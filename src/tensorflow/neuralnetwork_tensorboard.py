#!/usr/bin/env python
# encoding: utf-8

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


dataset = np.array([
((-0.4, 0.3), 0),
((-0.3, -0.1), 0),
((-0.2, 0.4), 0),
((-0.1, 0.1), 0),
((0.6, -0.5), 0), #non-linear point
((0.8, 0.7), 1),
((0.9, -0.5), 1),
((0.7, -0.9), 1),
((0.8, 0.2), 1),
((0.4, -0.6), 1)])


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
         with tf.name_scope('weights'):
             Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
             tf.histogram_summary(layer_name + '/weights', Weights)
         with tf.name_scope('bias'):
             biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
             tf.histogram_summary(layer_name + '/biases', biases)
         with tf.name_scope('Wx_plus_b'):
             Wx_plus_b = tf.matmul(inputs, Weights) + biases
         if activation_function is None:
             outputs = Wx_plus_b
         else:
             outputs = activation_function(Wx_plus_b)
         tf.histogram_summary(layer_name + '/outputs', outputs)
         return outputs


x_data = np.matrix([x for x,y in dataset])
y_data = np.matrix([y for x,y in dataset]).T

# define place hokader inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 2], name='x_data')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_data')

l1 = add_layer(xs, 2, 10, n_layer=1, activation_function=tf.sigmoid)
o = add_layer(l1, 10, 1, n_layer=2, activation_function=tf.sigmoid)
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(ys * -tf.log(o) + (1-ys) * -tf.log(1-o), name='cross_entropy')
    tf.scalar_summary('loss', cross_entropy)
with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


with tf.Session() as sess:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("logs/", sess.graph)
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(1000):
        sess.run(train, feed_dict={xs:x_data, ys:y_data})
        if i % 50 == 0:
            result = sess.run(merged,
                              feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(result, i)

#tensorboard  --logdir='logs/'
