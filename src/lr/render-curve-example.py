import numpy as np
import matplotlib.pyplot as plt
from sympy import *


X = np.linspace(0, 5, 300)
f = lambda x: x**2 - 2*x
F = np.vectorize(f)
Y = F(X)


plt.plot(X, Y, 'b')
plt.plot([1],[-1],'ro')
plt.plot([0, 5],[-1, -1],'black')
plt.show()

