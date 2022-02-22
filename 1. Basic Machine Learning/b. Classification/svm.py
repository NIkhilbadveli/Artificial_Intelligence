import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts, train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.svm import SVC


# See the cost equation here: https://miro.medium.com/max/826/1*a-xLx3Ht-Zk7mZddNy53tw.png
def cost(W, X, Y):
    N = X.shape[0]
    dist = 1 - Y * np.dot(X, W)
    dist[dist < 0] = 0
    hinge_loss = C * np.sum(dist) / N
    return 1 / 2 * np.sum(W ** 2) + hinge_loss


def gradient(W, X, Y):
    # Gradient differs for each row of the data
    N = X.shape[0]
    gr = np.zeros(W.shape)
    for j in range(N):
        dist = 1 - Y[j] * np.dot(X[j], W)
        if dist <= 0:
            gr += W
        else:
            gr += W - (C * Y[j] * X[j]).reshape(W.shape)

    return gr / N


def update_weights(lr, W, GR):
    return W - lr * GR


def predict(X, W):
    return np.sign(np.dot(X, W))


def fit_using_gradient_descent(X, Y, lr, n_iter):
    a = np.zeros((X.shape[1], 1))

    for i in range(n_iter):
        cost_history.append(cost(a, X, Y))
        a = update_weights(lr, a, gradient(a, X, Y))
    return a


C = 1  # Larger the C, closer it is to Maximal-Margin classifier
cost_history = []

cancer = datasets.load_breast_cancer()
x_data = cancer.data
y_data = cancer.target

X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data)

trainY = Y_train.reshape(Y_train.shape[0], 1)
testY = Y_test.reshape(Y_test.shape[0], 1)

iterations = 300
learning_rate = 0.01

A = fit_using_gradient_descent(X_train, Y_train, learning_rate, iterations)
y_pred = predict(X_test, A)

# Todo investigate why my accuracy is so low?
print('Accuracy with own implementation is :- ', 100 * accuracy_score(Y_test, y_pred))

# plt.plot(cost_history)
# plt.show()

model = SVC(kernel='linear')
model.fit(X_train, Y_train)

y_pred1 = model.predict(X_test)
print("Accuracy with sklearn's implementation is :- ", 100 * accuracy_score(Y_test, y_pred1))
