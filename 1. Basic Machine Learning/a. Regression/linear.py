import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

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


def fit_using_gradient_descent(X, Y, lr, n_iter):
    a = np.zeros((X.shape[1], 1))

    for i in range(n_iter):
        y1 = predict(X, a)
        cost_history.append(cost(Y, y1))
        avg_error_history.append(avg_error_perc(Y, y1))
        a = update_weights(lr, a, gradient(X, Y, y1))
    return a


def find_analytical_solution(X, Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y


def fit_using_sklearn(X, Y):
    # Try to do the same using sklearn
    rg = LinearRegression()
    rg.fit(X, Y)
    return rg.coef_, rg.intercept_, rg


df = pd.read_csv('linear_dataset.csv')

# Some data visualization
# plt.hist(df['GrLivArea'])
# sns.displot(df['SalePrice'])
# plt.show()

x = df[['OverallQual', 'GrLivArea', 'GarageCars']]
y = df['SalePrice'].to_numpy()

x = (x - x.mean()) / x.std()  # This is rescaling to Gaussian standard distribution
x = np.c_[np.ones(x.shape[0]), x]  # what does this expression do?

cost_history = []
avg_error_history = []

split_factor = 0.8
split_index = int(len(x) * split_factor)
trainX, testX = x[:split_index], x[split_index:]
trainY, testY = y[:split_index], y[split_index:]

trainY = trainY.reshape(trainY.shape[0], 1)
testY = testY.reshape(testY.shape[0], 1)

iterations = 300
learning_rate = 0.01

A = fit_using_gradient_descent(trainX, trainY, learning_rate, iterations)
B = find_analytical_solution(trainX, trainY)
C, D, regressor = fit_using_sklearn(trainX, trainY)

# Comparing average absolute error of the 3 methods.
print(avg_error_perc(testY, predict(testX, A)))  # Error is 17.14% - achieved solution close to analytical one.
print(avg_error_perc(testY, predict(testX, B)))  # Error is 17.16%
print(avg_error_perc(testY, regressor.predict(testX)))  # Error is 17.16%

# See the plot of cost getting reduced.
# plt.plot(avg_error_history)
# plt.plot(cost_history)
# plt.show()

# See how the predictions compare to the actual output.
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
