
# モジュールをインポート
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

# =============================================================================
# Default settings
# =============================================================================
# Supress scientific notations
np.set_printoptions(suppress=True)

# Define a random seed
random_seed = 123
np.random.seed(random_seed)

# =============================================================================
# Import samples
# =============================================================================
iris = datasets.load_iris()
X = iris.data[:, [0,3]]
y = iris.target

# =============================================================================
# Separate samples
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_seed)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.25, random_state = random_seed)

# =============================================================================
# Data standardization
# =============================================================================
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_valid_std = stdsc.transform(X_valid)
X_test_std = stdsc.transform(X_valid)

# =============================================================================
# Weight initialization
# =============================================================================
weight = np.random.normal(scale = 0.01, size = (X.shape[1] + 1, len(np.unique(y))))

# =============================================================================
# One hot encoder
# =============================================================================
class_labels = int(y.max() + 1)
def one_hot_encoder(y):
    a = np.zeros((len(y), class_labels))
    for idx, i in enumerate(y):
        a[idx, int(i)] = 1
    return a

y_train_coded = one_hot_encoder(y_train)
y_valid_coded = one_hot_encoder(y_valid)
y_test_coded= one_hot_encoder(y_test)

# =============================================================================
# Softmax function
# =============================================================================
def softmax(z):
    prob = np.exp(z)
    prob = prob / np.exp(z).sum(axis = 1).reshape(-1, 1)
    return prob

# =============================================================================
# Net input function
# =============================================================================
def net_input(X, weight):
    return X.dot(weight[1:]) + weight[0]
# =============================================================================
# Defining variables
# =============================================================================

# Number of epochs
epochs = 100

# Learning rate
learning_rate = 0.01

# Regularization Lambda
cost_lambda = 0

# =============================================================================
# Predict function
# =============================================================================
def predict(activation):
    y_predicted = activation.argmax(axis = 1)
    return y_predicted

# =============================================================================
# Full predict function
# =============================================================================
def full_predict(X, weight):
    z = net_input(X, weight)
    activation = softmax(z)
    y_predicted = activation.argmax(axis = 1)
    return y_predicted
    
# =============================================================================
# Compute cost function
# =============================================================================
def compute_cost(X, y):
    regularization = (cost_lambda * (weight[1:]**2).sum()) / (2 * X.shape[1])
    z = net_input(X, weight)
    activation = softmax(z)
    y_predicted = predict(activation)
    y_pred_enc = one_hot_encoder(y_predicted)
    cost = -np.sum(np.log(activation) * y_pred_enc) + regularization
    return cost

# =============================================================================
# Training function
# =============================================================================
def train(X, y, weight, epochs, learning_rate, cost_lambda):
    cost_array = []
    for epoch in range(epochs):
        z = net_input(X, weight) 
        activation = softmax(z)
        cost = compute_cost(X, y)
        cost_array.append(cost) 
        diff = activation - y_train_coded
        grad = np.dot(X.T, diff)
        
        weight[1:] -= learning_rate * (grad + cost_lambda * weight[1:] / X.shape[1])
        weight[0] -= learning_rate * np.sum(diff, axis = 0)
    return weight, cost_array

# =============================================================================
# Train the classifier
# =============================================================================
learned_weights, cost_array = train(X_train_std, y_train, weight, epochs, learning_rate, cost_lambda)

# =============================================================================
# Plot the cost-epoch graph
# =============================================================================
plt.plot(range(len(cost_array)), cost_array)
plt.show()

# =============================================================================
# Evaluation
# =============================================================================
y_pred = full_predict(X_train_std, learned_weights)
print("")
print("テストデータでの性能：")
conf_mat = confusion_matrix(y_train, y_pred)
print("混同行列：")
print(conf_mat)
accuracy = accuracy_score(y_train, y_pred)
print("精度:")
print(accuracy)

# =============================================================================
# Evaluation of Softmax using LogisticRegression from sklearn
# =============================================================================
classifier = LogisticRegression(C = 10)
classifier.fit(X_train_std, y_train)

y_pred = classifier.predict(X_train_std)
print("")
print("テストデータでの性能：")
conf_mat = confusion_matrix(y_train, y_pred)
print("混同行列：")
print(conf_mat)
accuracy = accuracy_score(y_train, y_pred)
print("精度:")
print(accuracy)


# Data plot

def plot_decision_regions(X, y, weight, resolution = 0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    print(np.array([xx1.ravel(), xx2.ravel()]).T.shape)
    Z = full_predict(np.array([xx1.ravel(), xx2.ravel()]).T, weight)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha = 0.8, c = colors[idx], marker = markers[idx], label = cl, edgecolor = 'black')

def plot_decision_regions_sklearn(X, y, classifier, resolution = 0.02):
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
        
def plot_decision_regions_poly(X, y, weight, resolution = 0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    X_poly = poly.transform(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = full_predict(X_poly, weight)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha = 0.8, c = colors[idx], marker = markers[idx], label = cl, edgecolor = 'black')


plot_decision_regions(X_train_std, y_train, weight)
plt.show()

plot_decision_regions_sklearn(X_train_std, y_train, classifier = classifier)
plt.show()

# =============================================================================
# Polynomial models
# =============================================================================

for i in range(2, 6):
    
    poly = PolynomialFeatures(i, include_bias = False)
    X_train_std_poly = poly.fit_transform(X_train_std)
    X_valid_std_poly = poly.transform(X_train_std)
    weight = np.random.normal(scale = 0.01, size = (X_train_std_poly.shape[1] + 1, len(np.unique(y))))
    
    learned_weights_train, cost_array_train = train(X_train_std_poly, y_train, weight, epochs, learning_rate, cost_lambda)
    
    
    y_pred = full_predict(X_train_std_poly, learned_weights_train)
    print("")
    print("テストデータでの性能：")
    conf_mat = confusion_matrix(y_train, y_pred)
    print("混同行列：")
    print(conf_mat)
    accuracy = accuracy_score(y_train, y_pred)
    print("精度:")
    print(accuracy)
    plot_decision_regions_poly(X_train_std_poly, y_train, weight)
    plt.show()