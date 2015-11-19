import numpy as np
import matplotlib.pyplot as plt

#Target function line
X = np.linspace(0, 10)
f = lambda x: x #y=x
F = np.vectorize(f)
Y = F(X)

#random data by F(X) + random residual(upper bound=2)
num = 15 #number of data
random_sign = np.vectorize(lambda x: x if np.random.sample() > 0.5 else -x)
data_X = np.linspace(1, 9, num)
data_Y = random_sign(np.random.sample(num) * 2) + F(data_X)

#using sympy
from sympy import *

def linear_regression(X, Y):
    a, b = symbols('a b')
    residual = 0
    for i in range(num):
        residual += (Y[i] - (a * X[i] + b)) ** 2

    print expand(residual)
    f1 = diff(residual, a)
    f2 = diff(residual, b)
    print f1
    print f2
    res = solve([f1, f2], [a, b])
    return res[a], res[b]

a, b = linear_regression(data_X, data_Y)
print a,b

LR_X = X
h = lambda x: a*x + b
H = np.vectorize(h)
LR_Y = H(LR_X)

#render residual
#for i in range(num):
#    plt.plot([data_X[i], data_X[i]], [data_Y[i], h(data_X[i])], 'black')

#Simple line regression: y=ax+b
#http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
#A = np.vstack([data_X, np.ones(len(data_X))]).T
#a, b = np.linalg.lstsq(A, data_Y)[0]


plt.plot(X, Y, 'b') # render blue line
plt.plot(LR_X, LR_Y, 'g') # render green line
plt.plot(data_X, data_Y, 'ro')
plt.show()
