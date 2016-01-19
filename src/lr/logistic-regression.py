#!/usr/bin/env python
# encoding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import math

#網路上找的dataset 可以線性分割

dataset = np.array([
((1, -0.4, 0.3), -1),
((1, -0.3, -0.1), -1),
((1, -0.2, 0.4), -1),
((1, -0.1, 0.1), -1),
((1, 0.6, -0.5), -1), #non-linear point
((1, 0.8, 0.7), 1),
((1, 0.9, -0.5), 1),
((1, 0.7, -0.9), 1),
((1, 0.8, 0.2), 1),
((1, 0.4, -0.6), 1)])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient(dataset, w):
    g = 0
    for x,y in dataset:
        x = np.array(x)
        g += sigmoid(-y * w.T.dot(x)) * (-y * x)
    return g / len(dataset)

def logistic(dataset):
    w = np.zeros(3)
    limit = 10
    eta = 1
    for i in range(limit):
        w = w - eta * gradient(dataset, w)
        eta *= 0.9
    return w
#執行

w = logistic(dataset)
#畫圖

ps = [v[0] for v in dataset]
fig = plt.figure()
ax1 = fig.add_subplot(111)
#dataset前半後半已經分割好 直接畫就是

ax1.scatter([v[1] for v in ps[:5]], [v[2] for v in ps[:5]], s=10, c='b', marker="o", label='O')
ax1.scatter([v[1] for v in ps[5:]], [v[2] for v in ps[5:]], s=10, c='r', marker="x", label='X')
l = np.linspace(-2,2)
a,b = -w[1]/w[2], -w[0]/w[2]
ax1.plot(l, a*l + b, 'b-')
plt.legend(loc='upper left');
plt.show()
