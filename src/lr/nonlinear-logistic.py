import numpy as np
import matplotlib.pyplot as plt
import itertools


###
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient(dataset, w):
    g = 0
    for x,y in dataset:
        x = np.array(x)
        g += sigmoid(-y * w.T.dot(x)) * (-y * x)
    return g / len(dataset)

def logistic(dataset):
    w = np.zeros(len(dataset[0][0]))
    limit = 50
    eta = 1
    for i in range(limit):
        w = w - eta * gradient(dataset, w)
        eta *= 0.9
    return w
###


def tf(X):
    result = []
    for x in X:
        result.append((1, x[0] ** 2 +  x[1] ** 2 - 5 ))
    return result

class1 = [(1.1,2.1), (-1.2,-2.2), (1.3,-2.3), (-1.4,2.4)]
class2 = [(-10.1, -10.1), (10.2, 10.2), (10.3, -10.3), (-10.4, 10.4)]

c1_dataset = tf(class1)
c2_dataset = tf(class2)
training_set = [(v, 1) for v in c1_dataset]
training_set += [(v, -1) for v in c2_dataset]

W = logistic(training_set)

plt.plot([v[0] for v in class1 ], [v[1] for v in class1 ],'bo')
plt.plot([v[0] for v in class2 ], [v[1] for v in class2 ],'rx')
plt.xlim(-12, 12)
plt.ylim(-12, 12)
plt.show()

test_set = [[0, 0],[0, 10], [2, 0], [2.5, 0], [0, 1.5], [0, 2.5]]
for data in test_set:
    plt.plot([data[0]], [data[1]],'k.')
plt.xlim(-12, 12)
plt.ylim(-12, 12)
plt.show()


W = np.array([W])
for data in test_set:
    label = ''
    feature = (np.array(tf([data])))
    if W.dot(feature.T) > 0:
        label = 'bo'
    else:
        label = 'rx'
    plt.plot([data[0]], [data[1]], label)

plt.xlim(-12, 12)
plt.ylim(-12, 12)
plt.show()
