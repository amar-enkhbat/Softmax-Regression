import matplotlib.pyplot as plt 
import numpy as np 

np.set_printoptions(suppress=True)

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, [1, 2]]
y = iris.target

# Data plot
for i in np.unique(y):
    plt.scatter(X[y == i, 0], X[y == i, 1], label = "Class" + str(i))
plt.show()

# Data standardization
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()

X_train = stdsc.fit_transform(X)

# サンプルとラベルを連結
train_test_split = np.c_[X_train, y]

# Initiate weights
rgen = np.random.RandomState(1)
weight = rgen.normal(scale = 0.1, size = (X.shape[1] + 1, len(np.unique(y))))
rgen.shuffle(train_test_split)
X_train = train_test_split[:, :X.shape[1]]
y = train_test_split[:, X.shape[1]]

def one_hot_encoder(y):
    a = np.zeros((len(y), int(y.max())+1))
    for idx, i in enumerate(y):
        a[idx, int(i)] = 1
    return a

y_coded = one_hot_encoder(y)
#X_train = np.column_stack((np.ones(len(X)), X_train))

#def softmax(X, weight):
#    prob = []
#    for j in y:
#        net_input = np.exp(weight[1:, int(j)].dot(X) + weight[0, int(j)])
#        prob.append(net_input)
#    prob = prob / np.exp(weight[1:].T.dot(X) + weight[0, int(j)]).sum()
##    print(prob)
#    return prob
#
#def activate(X, weight):
#    prob = []
#    for a in X:
#        prob.append(softmax(a, weight))
#    return np.array(prob)
def softmax(X, weight):
    net_input = X.dot(weight[1:]) + weight[0]
    prob = np.exp(net_input)
    prob = prob / np.exp(net_input).sum(axis = 1).reshape(-1, 1)
    return prob

# =============================================================================
# 
# epoch = 1
# eta = 0.01
# cost_series = []
# cost_lambda = 0
# for epochs in range(epoch):
#     
#     z = activate(X_train, weight)
#     y_pred = z.argmax(axis = 1)
#     y_pred_enc = one_hot_encoder(y_pred)
#     regularization = (cost_lambda * (weight[1:]**2).sum()) / (2 * X_train.shape[1])
#     cost = -np.sum(np.log(activate(X_train, weight)) * y_pred_enc) + regularization
# #    print(cost)
#     cost_series.append(cost) 
#     
#     diff = activate(X_train, weight) - y_coded
#     grad = np.dot(X_train.T, diff)
#     
#     weight[1:] -= eta * (grad + cost_lambda * weight[1:] / X_train.shape[1])
#     weight[0] -= eta*np.sum(diff, axis = 0)
# #    weight[0] -= eta * np.sum(diff, axis = 0)
#     if len(cost_series) > 2:
#         if cost_series[-2] - cost_series[-1] < 0.001:
#             break
# 
# plt.plot(range(len(cost_series)), cost_series)
# plt.show()
# 
# 
# z = np.flip(np.unique(z, axis = 1), axis = 1)
# y_pred = z.argmax(axis = 1)
# 
# from sklearn.metrics import confusion_matrix, accuracy_score
# print("")
# print("テストデータでの性能：")
# conf_mat = confusion_matrix(y, y_pred)
# print("混同行列：")
# print(conf_mat)
# accuracy = accuracy_score(y, y_pred)
# print("精度:")
# print(accuracy)
# =============================================================================

        
