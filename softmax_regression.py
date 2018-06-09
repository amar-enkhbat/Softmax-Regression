
# モジュールをインポート
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

np.set_printoptions(suppress=True)
#　ランダムシード
rgen = np.random.RandomState(2)

# サンプルロード
iris = datasets.load_iris()
X = iris.data[:, [0,3]]
y = iris.target

# クラスラベル数
class_labels = int(y.max() + 1)

# Train, test, split
from sklearn.model_selection import train_test_split
# サンプルをシャフル
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.25, random_state = 2)

# Data standardization
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_valid_std = stdsc.transform(X_valid)
X_test_std = stdsc.transform(X_valid)


# 重みを初期化
weight = rgen.normal(scale = 0.1, size = (X.shape[1] + 1, len(np.unique(y))))

# 符号化関数
def one_hot_encoder(y):
    a = np.zeros((len(y), class_labels))
    for idx, i in enumerate(y):
        a[idx, int(i)] = 1
    return a

y_train_coded = one_hot_encoder(y_train)
y_valid_coded = one_hot_encoder(y_valid)
y_test_coded= one_hot_encoder(y_test)


# Softmax function
def softmax(X, weight):
    net_input = X.dot(weight[1:]) + weight[0]
    prob = np.exp(net_input)
    prob = prob / np.exp(net_input).sum(axis = 1).reshape(-1, 1)
    return prob

# Number of epochs
epochs = 100

# Learning rate
learning_rate = 0.01
cost_array = []
cost_lambda = 0.1
for epoch in range(epochs):
    z = softmax(X_train_std, weight)
    y_pred = z.argmax(axis = 1)
    y_pred_enc = one_hot_encoder(y_pred)
    regularization = (cost_lambda * (weight[1:]**2).sum()) / (2 * X_train_std.shape[1])
    cost = -np.sum(np.log(z) * y_pred_enc) + regularization
#    print(cost)
    cost_array.append(cost) 

    diff = z - y_train_coded
    grad = np.dot(X_train_std.T, diff)
    
    weight[1:] -= learning_rate * (grad + cost_lambda * weight[1:] / X_train.shape[1])
    weight[0] -= learning_rate * np.sum(diff, axis = 0)

    
    if len(cost_array) > 2:
        if cost_array[-2] - cost_array[-1] < 0.0001:
            break
        
plt.plot(range(len(cost_array)), cost_array)
plt.show()

z = softmax(X_train_std, weight)
y_pred = z.argmax(axis = 1)
 
from sklearn.metrics import confusion_matrix, accuracy_score
print("")
print("テストデータでの性能：")
conf_mat = confusion_matrix(y_train, y_pred)
print("混同行列：")
print(conf_mat)
accuracy = accuracy_score(y_train, y_pred)
print("精度:")
print(accuracy)


from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(C = 10)
classifier.fit(X_train_std, y_train)
y_pred = classifier.predict(X_train_std)

from sklearn.metrics import confusion_matrix, accuracy_score
print("")
print("テストデータでの性能：")
conf_mat = confusion_matrix(y_train, y_pred)
print("混同行列：")
print(conf_mat)
accuracy = accuracy_score(y_train, y_pred)
print("精度:")
print(accuracy)


# Data plot

from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, weight, resolution = 0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = softmax(np.array([xx1.ravel(), xx2.ravel()]).T, weight)
    Z = Z.argmax(axis = 1)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha = 0.8, c = colors[idx], marker = markers[idx], label = cl, edgecolor = 'black')

def plot_decision_regions1(X, y, classifier, resolution = 0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha = 0.8, c = colors[idx], marker = markers[idx], label = cl, edgecolor = 'black')


plot_decision_regions(X_train_std, y_train, weight)
plt.show()

plot_decision_regions1(X_train_std, y_train, classifier = classifier)
plt.show()