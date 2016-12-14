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


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def draw(plt, ax, sess, o):
    x_dataset =[]
    o_dataset = []
    line = np.linspace(-1,1)
    for x in line:
        for y in line:
            res = sess.run(o, feed_dict={xs:[[x, y]], keep_prob: 1})[0]
            if res > 0.5:
                o_dataset.append([x, y])
            else:
                x_dataset.append([x, y])
    ax.scatter([v[0] for v in o_dataset], [v[1] for v in o_dataset], s=4, c='b', marker="o", label='O')
    ax.scatter([v[0] for v in x_dataset], [v[1] for v in x_dataset], s=4, c='r', marker="x", label='X')
    plt.pause(0.1)


x_data = np.matrix([x for x,y in dataset])
y_data = np.matrix([y for x,y in dataset]).T

keep_prob = tf.placeholder(tf.float32) # For dropout 
xs = tf.placeholder(tf.float32, [None, 2])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 2, 10, activation_function=tf.sigmoid)
o = add_layer(l1, 10, 1, activation_function=tf.sigmoid)
cross_entropy = tf.reduce_mean(ys * -tf.log(o) + (1-ys) * -tf.log(1-o))
train = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.ion()
plt.xlim(-1, 1)
plt.ylim(-1, 1)
ax.scatter([v.item(0) for v in x_data[:5]], [v.item(1) for v in x_data[:5]], s=500, c='r', marker="x", label='X')
ax.scatter([v.item(0) for v in x_data[5:]], [v.item(1) for v in x_data[5:]], s=500, c='b', marker="o", label='O')
plt.show()

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(10000):
        sess.run(train, feed_dict={xs:x_data, ys:y_data, keep_prob: 0.5}) # Use 50% tensor for trainning
        if i % 500 == 0:
            print sess.run(cross_entropy, feed_dict={xs:x_data, ys:y_data, keep_prob: 1}) # Use all tensor for output
            draw(plt, ax, sess, o)
