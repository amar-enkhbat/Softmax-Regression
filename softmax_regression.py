import matplotlib.pyplot as plt 
import numpy as np 

t = np.arange(-3, 3, 0.1)

def func1(x):
    return np.exp(x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

plt.plot(t, func1(t), label = "exp(X)")
plt.plot(t, sigmoid(t), label = "Sigmoid")
plt.xlabel("X")
plt.ylabel("exp(X)")
plt.legend()
plt.show()