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


df = pd.read_csv('linear_dataset.csv')
x = df[['OverallQual', 'GrLivArea', 'GarageCars']]
y = df['SalePrice'].to_numpy()

x = (x - x.mean()) / x.std()  # This is rescaling to Gaussian standard distribution
x = np.c_[np.ones(x.shape[0]), x]  # what does this expression do?

# plt.hist(df['GrLivArea'])
# sns.displot(df['SalePrice'])
# plt.show()

split_factor = 0.8
split_index = int(len(x) * split_factor)
trainX, testX = x[:split_index], x[split_index:]
trainY, testY = y[:split_index], y[split_index:]

trainY = trainY.reshape(trainY.shape[0], 1)
testY = testY.reshape(testY.shape[0], 1)

n_iter = 300
learning_rate = 0.01
A = np.zeros((trainX.shape[1], 1))
cost_history = []
avg_error_history = []

for i in range(n_iter):
    y1 = predict(trainX, A)
    cost_history.append(cost(trainY, y1))
    avg_error_history.append(avg_error_perc(trainY, y1))
    A = update_weights(learning_rate, A, gradient(trainX, trainY, y1))

# plt.plot(avg_error_history)
# plt.show()

# print(cost_history[-1])
# print(cost(predict(testX, A), testY))

# p_x = range(len(testY))
# plt.scatter(p_x, testY)
# plt.scatter(p_x, predict(testX, A))
# plt.plot(error_perc_array(testY, predict(testX, A)))
# plt.show()

# Learnings and things to note :-
# Since this is not a classification task, the only way to know how good the model is by looking at the cost.
# It's not intuitive because just a raw number doesn't tell anything about the accuracy.
# There should be a better way to know, so that there's apples-to-apples comparison.
# I just realized that I can simply calculate how far the predictions are off for each value and then take average.
# For the above model, this number (Average Absolute Error) is somewhere around 18%.
# I believe I can use the same metric for comparing other regression methods.
# When you have more features, pick features after seeing the data carefully and weed out any that have high correlation
# Also, deal with missing data or just ignore those columns. Plus don't forget to rescale the data.
# I chose Mean Squared Error as the cost and Gradient Descent as the optimizer.

# Try to do the same using sklearn
# Also, try the analytical solution of A = np.linalg.inv(X.T@X) @ X.T @ Y where @ denotes dot product
