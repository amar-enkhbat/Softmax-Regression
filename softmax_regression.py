import matplotlib.pyplot as plt 
import numpy as np 

np.set_printoptions(suppress=True)

from sklearn import datasets
wine = datasets.load_wine()
X = wine.data[:, [0, 3]]
y = wine.target

class_labels = int(y.max() + 1)
# Data plot
#for i in np.unique(y):
#    plt.scatter(X[y == i, 0], X[y == i, 1], label = "Class" + str(i))
#plt.show()
# =============================================================================
# feature_1 = np.array([2, 5, 8, 4], dtype = float)
# feature_2 = np.array([5, 1, 6, 9], dtype = float)
# 
# # sample
# X = np.column_stack((feature_1, feature_2))
# 
# # label
# y = np.arange(len(feature_1))
# class_labels = int(y.max() + 1)
# =============================================================================
# Data standardization
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()

X_train = stdsc.fit_transform(X)

# サンプルとラベルを連結
train_test_split = np.c_[X_train, y]

# Initiate weights
rgen = np.random.RandomState(1)
weight = rgen.normal(scale = 1, size = (X.shape[1] + 1, len(np.unique(y))))
rgen.shuffle(train_test_split)
X_train = train_test_split[:, :X.shape[1]]
y = train_test_split[:, X.shape[1]]

def one_hot_encoder(y):
    a = np.zeros((len(y), class_labels))
    for idx, i in enumerate(y):
        a[idx, int(i)] = 1
    return a

y_coded = one_hot_encoder(y)

def softmax(X, weight):
    net_input = X.dot(weight[1:]) + weight[0]
    prob = np.exp(net_input)
    prob = prob / np.exp(net_input).sum(axis = 1).reshape(-1, 1)
    return prob



epoch = 10000
eta = 0.01
cost_series = []
cost_lambda = 0.01
for epochs in range(epoch):
    z = softmax(X_train, weight)
    y_pred = z.argmax(axis = 1)
    y_pred_enc = one_hot_encoder(y_pred)
    regularization = (cost_lambda * (weight[1:]**2).sum()) / (2 * X_train.shape[1])
    cost = -np.sum(np.log(z) * y_pred_enc) + regularization
#    print(cost)
    cost_series.append(cost) 

    diff = z - y_coded
    grad = np.dot(X_train.T, diff)
    
    weight[1:] -= eta * (grad + cost_lambda * weight[1:] / X_train.shape[1])
    weight[0] -= eta*np.sum(diff, axis = 0)

    if len(cost_series) > 2:
        if cost_series[-2] - cost_series[-1] < 0.0001:
            break

plt.plot(range(len(cost_series)), cost_series)
plt.show()

y_pred = z.argmax(axis = 1)
 
from sklearn.metrics import confusion_matrix, accuracy_score
print("")
print("テストデータでの性能：")
conf_mat = confusion_matrix(y, y_pred)
print("混同行列：")
print(conf_mat)
accuracy = accuracy_score(y, y_pred)
print("精度:")
print(accuracy)


from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(C = 100)
classifier.fit(X_train, y)
y_pred = classifier.predict(X_train)

from sklearn.metrics import confusion_matrix, accuracy_score
print("")
print("テストデータでの性能：")
conf_mat = confusion_matrix(y, y_pred)
print("混同行列：")
print(conf_mat)
accuracy = accuracy_score(y, y_pred)
print("精度:")
print(accuracy)


# Data plot
t = np.arange(-3, 4, 0.1)
for i in np.unique(y):
    i = int(i)
    plt.scatter(X_train[y == i, 0], X_train[y == i, 1], label = "Class" + str(i))
    plt.plot(t, -(weight[0, i] + weight[1, i] * t)/weight[2, i])
plt.xlim(-3, 3)
plt.ylim(-2, 2)
plt.show()
