import matplotlib.pyplot as plt 
import numpy as np 

np.set_printoptions(suppress=True)
# t = np.arange(-3, 3, 0.1)

# def func1(x):
#     return np.exp(x)

# def sigmoid(x):
#     return 1/(1 + np.exp(-x))

# plt.plot(t, func1(t), label = "exp(X)")
# plt.plot(t, sigmoid(t), label = "Sigmoid")
# plt.xlabel("X")
# plt.ylabel("exp(X)")
# plt.legend()
# plt.show()

feature_1 = np.array([2, 5, 8, 4], dtype = float)
feature_2 = np.array([5, 1, 6, 9], dtype = float)

# sample
X = np.column_stack((feature_1, feature_2))

# label
y = np.arange(0, len(feature_1))
y_coded = np.array([[1, 0, 0, 0], 
                    [0, 1, 0, 0], 
                    [0, 0, 1, 0], 
                    [0, 0, 0, 1]])

#from sklearn import datasets
#iris = datasets.load_iris()
#X = iris.data[:, :2]
#y = iris.target
#y_coded = one_hot_encoder(y)
# Plot data points
""" for i in y:
    plt.scatter(feature_1[y == i], feature_2[y == i], label = "Label " + str(i))
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.legend()
plt.show() """

# Data standardization
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()

X_train = stdsc.fit_transform(X)

# Initiate weights
rgen = np.random.RandomState(1)
weight = rgen.normal(scale = 1, size = (X.shape[1] + 1, len(y)))

#X_train = np.column_stack((np.ones(len(X)), X_train))

def softmax(X, weight):
    prob = []
    for j in y:
        net_input = np.exp(X.dot(weight[1:, j]) + weight[0, j])
        prob.append(net_input)
    prob = prob / np.exp(X.dot(weight[1:]) + weight[0, j]).sum()
#    print(prob)
    return prob

def activate(X, weight):
    prob = []
    for a in X:
        prob.append(softmax(a, weight))
    return np.array(prob)

def one_hot_encoder(y):
    a = np.zeros((len(y), len(y)))
    for idx, i in enumerate(y):
        a[idx, i] = 1
    return a

epoch = 1000
eta = 0.01
cost_series = []

for epochs in range(epoch):
    
    z = activate(X_train, weight)
    y_pred = z.argmax(axis = 1)
    y_pred_enc = one_hot_encoder(y_pred)
    
    cost = -np.sum(np.log(activate(X_train, weight)) * y_pred_enc)
#    print(cost)
    cost_series.append(cost) 
    
    diff = activate(X_train, weight) - y_coded
    grad = np.dot(X_train.T, diff)
    
    weight[1:] -= eta * grad
    weight[0] -= eta*np.sum(diff, axis = 0)
    print(weight)
#    weight[0] -= eta * np.sum(diff, axis = 0)

plt.plot(range(len(cost_series)), cost_series)
plt.show()


t = np.arange(-2, 2, 0.1)
feature_1 = X_train[:, 0]
feature_2 = X_train[:, 1]
for i in range(len(y)):
    plt.scatter(feature_1[y == i], feature_2[y == i])
    plt.plot(t, (weight[0, i] + t * weight[1, i]) / (-weight[2, i]))

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()
        
