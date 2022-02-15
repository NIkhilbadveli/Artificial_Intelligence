import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()


def cost(Y, Y1):
    return ((Y - Y1) ** 2).sum() / len(Y)


def gradient(X, Y, Y1):
    return 2 / len(Y) * np.dot(X.T, Y1 - Y)


def update_weights(lr, W, GR):
    return W - lr * GR


def predict(X, W):
    return np.dot(X, W)


def avg_error_perc(Y, Y1):
    return (np.abs(Y - Y1) / Y).mean() * 100


def error_perc_array(Y, Y1):
    return ((Y - Y1) / Y) * 100


def poly_fit(X, Y, p, lr, n_iter):
    # p = degree of the polynomial
    t = X.copy()
    for i in range(2, p + 1):
        X = np.append(X, t ** i, axis=1)
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    A = np.zeros((X.shape[1], 1))

    cost_history = []
    avg_error_history = []

    for i in range(n_iter):
        y1 = predict(X, A)
        cost_history.append(cost(Y, y1))
        avg_error_history.append(avg_error_perc(Y, y1))
        A = update_weights(lr, A, gradient(X, Y, y1))

    print(avg_error_history[-1])
    # p_x = range(len(Y))
    # plt.scatter(p_x, Y)
    # plt.scatter(p_x, predict(X, A))
    # plt.plot(avg_error_history)
    # plt.show()
    return A


df = pd.read_csv('polynomial_dataset.csv')
x = df.loc[:, ~df.columns.isin(['Width', 'Species'])]
y = df['Width'].to_numpy()

# Tried fitting the linear data, not that much improvement.
# I guess it's important to see the shape of the data and choose the model accordingly
# df = pd.read_csv('linear_dataset.csv')
# x = df[['OverallQual', 'GrLivArea', 'GarageCars']]
# y = df['SalePrice'].to_numpy()
# y = y.reshape(y.shape[0], 1)

x = (x - x.mean()) / x.std()

poly_fit(X=x, Y=y, p=2, lr=0.01, n_iter=1000)
