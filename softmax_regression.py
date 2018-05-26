
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
X = iris.data[:, 0:2]
y = iris.target

# クラスラベル数
class_labels = int(y.max() + 1)

# Data standardization
stdsc = StandardScaler()
X_train = stdsc.fit_transform(X)

# サンプルとラベルを連結
train_test_split = np.c_[X_train, y]

# サンプルをシャフル
rgen.shuffle(train_test_split)
X_train = train_test_split[:, :X.shape[1]]
y = train_test_split[:, X.shape[1]]

# 重みを初期化
weight = rgen.normal(scale = 0.01, size = (X.shape[1] + 1, len(np.unique(y))))

# 符号化関数
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



epoch = 1000
eta = 0.01
cost_series = []
cost_lambda = 0.1
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

z = softmax(X_train, weight)
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

classifier = LogisticRegression(C = 10)
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


plot_decision_regions(X_train, y, weight)
plt.show()

plot_decision_regions1(X_train, y, classifier = classifier)
plt.show()