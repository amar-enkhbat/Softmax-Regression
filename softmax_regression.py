
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
from sklearn.model_selection import learning_curve

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
wine = datasets.load_iris()
X = wine.data[:, [0, 3]]
y = wine.target
number_of_classes = len(np.unique(y))

# =============================================================================
# Separate samples
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_seed)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = random_seed)



# =============================================================================
# Data standardization
# =============================================================================
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_valid_std = stdsc.transform(X_valid)
X_test_std = stdsc.transform(X_valid)

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
    log_c = np.max(z, axis = 1) * (-1)
    log_c = log_c.reshape(-1, 1)
    prob = np.exp(z + log_c)
    prob = prob / np.exp(z + log_c).sum(axis = 1).reshape(-1, 1)
    return np.clip(prob, 1e-15, 1-1e-15)
    
#    return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

# =============================================================================
# Net input function
# =============================================================================
def net_input(X, weight):
    return X.dot(weight[1:]) + weight[0]

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
def compute_cost(X, y, weight, cost_lambda):
    y_pred_enc = one_hot_encoder(y)
    z = net_input(X, weight)
    activation = softmax(z)
    cross_entropy = - np.sum(np.log(activation) * (y_pred_enc), axis = 1)
    regularization = cost_lambda * np.sum(weight[1:]**2) / 2
    cross_entropy = cross_entropy + regularization
    return np.mean(cross_entropy)

# =============================================================================
# Compute error
# =============================================================================

def compute_error(X, y, weight, cost_lambda):
    y = one_hot_encoder(y)
    z = net_input(X, weight)
    activation = softmax(z)
    regularization = cost_lambda * np.mean(weight[2:] ** 2)/2
    return np.mean((activation - y)**2)/2 + regularization

# =============================================================================
# Defining variables
# =============================================================================

# Number of epochs
epochs = 100

# Learning rate
learning_rate = 0.01

weight_scale = 1
# Regularization Lambda
cost_lambda = 10

# =============================================================================
# Training function
# =============================================================================
def train(X, y, epochs, learning_rate, cost_lambda):
    weight = np.random.normal(loc = 0, scale = weight_scale, size = (X.shape[1] + 1, number_of_classes))
    cost_array = []
    y_encoded = one_hot_encoder(y)
    for epoch in range(epochs):
        z = net_input(X, weight) 
        activation = softmax(z)
        diff = activation - y_encoded 
        grad = np.dot(X.T, diff)
        weight[1:] -= learning_rate * (grad + cost_lambda * weight[1:])
        weight[0] -= learning_rate * np.sum(diff, axis = 0)
        cost = compute_cost(X, y, weight, cost_lambda)
        cost_array.append(cost)
    return weight, cost_array

# =============================================================================
# Train the classifier
# =============================================================================
learned_weights, cost_array = train(X_train_std, y_train, epochs, learning_rate, cost_lambda)

# =============================================================================
# Plot the cost-epoch graph
# =============================================================================
plt.plot(range(len(cost_array)), cost_array)
plt.show()

# =============================================================================
# Evaluation
# =============================================================================
#y_pred = full_predict(X_train_std, learned_weights)
#print("")
#print("テストデータでの性能：")
#conf_mat = confusion_matrix(y_train, y_pred)
#print("混同行列：")
#print(conf_mat)
#accuracy = accuracy_score(y_train, y_pred)
#print("精度:")
#print(accuracy)

# =============================================================================
# Evaluation of Softmax using LogisticRegression from sklearn
# =============================================================================
#classifier = LogisticRegression(C = 10)
#classifier.fit(X_train_std, y_train)

#y_pred = classifier.predict(X_train_std)
#print("")
#print("テストデータでの性能：")
#conf_mat = confusion_matrix(y_train, y_pred)
#print("混同行列：")
#print(conf_mat)
#accuracy = accuracy_score(y_train, y_pred)
#print("精度:")
#print(accuracy)


# Data plot

def plot_decision_regions(X, y, weight, resolution = 0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
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

#
#plot_decision_regions(X_train_std, y_train, learned_weights)
#plt.show()
#
#plot_decision_regions_sklearn(X_train_std, y_train, classifier = classifier)
#plt.show()



# =============================================================================
# Polynomial models
# =============================================================================

for i in range(1, 10):
    
    poly = PolynomialFeatures(i, include_bias = False)
    X_train_std_poly = poly.fit_transform(X_train_std)
    X_valid_std_poly = poly.transform(X_train_std)
#    weight = np.random.normal(scale = 0.01, size = (X_train_std_poly.shape[1] + 1, len(np.unique(y))))
    
    learned_weights_train, cost_array_train = train(X_train_std_poly, y_train, epochs, learning_rate, cost_lambda)
    
    
    y_pred = full_predict(X_train_std_poly, learned_weights_train)
    print("")
    print("テストデータでの性能：")
    conf_mat = confusion_matrix(y_train, y_pred)
    print("混同行列：")
    print(conf_mat)
    accuracy = accuracy_score(y_train, y_pred)
    print("精度:")
    print(accuracy)
    plot_decision_regions_poly(X_train_std_poly, y_train, learned_weights_train)
    plt.show()
        
# =============================================================================
# Learning curve plot
# =============================================================================
# m = X_train_std.shape[0]
# error_train = np.zeros(m-1)
# error_val = np.zeros(m-1)

# for i in range(1, m):
#     learned_weights, a = train(X_train_std[0:i, :], y_train[0:i], epochs, learning_rate, cost_lambda)
#     error_train[i-1] = compute_error(X_train_std[0:i, :], y_train[0:i], learned_weights, 0)
#     error_val[i-1]= compute_error(X_valid_std, y_valid, learned_weights, 0)

# plt.plot(range(m-1), error_train, label = "Training data")
# plt.plot(range(m-1), error_val, label = "Validation data")
# plt.title("Learning Curve")
# plt.xlabel("Number of data")
# plt.ylabel("Cost")
# plt.legend()
# plt.show()


# =============================================================================
# Validation plot
# =============================================================================
# lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 3000]

# error_train = []
# error_val = []

# for i in lambda_vec:
#     learned_weights, a = train(X_train_std, y_train, epochs, learning_rate, i)
#     error_train.append(compute_error(X_train_std, y_train, learned_weights, 0))
#     error_val.append(compute_error(X_valid_std, y_valid, learned_weights, 0))
    
# plt.plot(range(len(lambda_vec)), error_train, label = "Training data")
# plt.plot(range(len(lambda_vec)), error_val, label = "Validation data")
# plt.title("Learning Curve")
# plt.xlabel("Lambda")
# plt.ylabel("Cost")
# plt.legend()
# plt.show()

print(learned_weights_train.shape)
# for i in range(10):
#     plt.imshow(learned_weights_train[i].reshape(28, 28))