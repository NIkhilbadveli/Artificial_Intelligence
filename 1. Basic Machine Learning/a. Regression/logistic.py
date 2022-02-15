# Note: Logistic regression is used for binary classification tasks and not strictly comes under regression.
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sns.set()


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def cost(Y, Y1):
    return -np.mean(Y * np.log(Y1) + (1 - Y) * np.log(1 - Y1))


def gradient(X, Y, Y1):
    return 2 / len(Y) * np.dot(X.T, Y1 - Y)


def update_weights(lr, W, GR):
    return W - lr * GR


def predict(X, W):
    return sigmoid(np.dot(X, W))


def predict_1(preds):
    return np.array([1 if i > 0.5 else 0 for i in preds])


def accuracy(Y, Y1):
    return ((Y == Y1).sum()) * 100 / len(Y)


def fit_using_gradient_descent(X, Y, lr, n_iter):
    a = np.zeros((X.shape[1], 1))

    for i in range(n_iter):
        y1 = predict(X, a)
        loss = cost(Y, y1)
        cost_history.append(loss)
        a = update_weights(lr, a, gradient(X, Y, y1))
    return a


def plot_line(X, m, c, Y):
    plt.scatter(X, Y)
    plt.plot(X, sigmoid(m * X + c))
    plt.show()


cost_history = []

df = pd.read_csv('logistic_dataset.csv')
df = df[['Age', 'Fare', 'Pclass', 'Survived']].dropna()
x = df[['Age', 'Fare', 'Pclass']].to_numpy()
y = df['Survived'].to_numpy()

# plt.scatter(x, y)
# plt.show()

# x = (x - x.mean()) / x.std()  # This is rescaling to Gaussian standard distribution
x = np.c_[np.ones(x.shape[0]), x]  # what does this expression do?

y = y.reshape((y.shape[0], 1))

A = fit_using_gradient_descent(x, y, 0.001, 1000)
# plt.scatter(list(range(len(cost_history))), cost_history)
# plt.show()
# plot_line(x[:, 1], A[1], A[0], y)
y_hat = predict_1(predict(x, A))
print(accuracy(y, y_hat.reshape((y_hat.shape[0], 1))))

log_reg = LogisticRegression()
log_reg.fit(x, y)
y_pred = log_reg.predict(x)
print(accuracy_score(y, y_pred))
