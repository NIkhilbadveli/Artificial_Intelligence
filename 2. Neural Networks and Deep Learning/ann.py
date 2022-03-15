# Code a simple ann (feed forward network) with sgd and backpropagation
# Take a binary classification task
import random

import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HiddenLayer:
    def __init__(self, num_neurons):
        # prev_layer_size will be set by the NeuralNetwork class
        self.dA = None
        self.z = None
        self.A = None
        self.input_data = None
        self.num_neurons = num_neurons
        self.prev_layer_size = 0
        self.is_output_layer = False

    def init_weights(self):
        # A is the matrix that includes both weights and biases
        self.A = np.random.rand(self.num_neurons, self.prev_layer_size + 1)
        self.z = np.zeros((self.num_neurons, 1))
        self.dA = np.zeros(self.A.shape)

    def feed_forward(self, input_data):
        # Here input_data should be a single sample of the shape (prev_layer_size x 1)
        input_data = np.append(input_data, 1).reshape(-1, 1)
        self.input_data = input_data
        self.z = self.A @ input_data
        return self.sigmoid(self.z)

    def back_prop(self, delta):
        self.input_data = self.input_data.reshape(-1, 1)
        self.dA += (self.input_data @ delta.T).T
        return self.dA

    def update_A(self, eta, n):
        self.A = self.A - eta * (self.dA / n)

    # Todo try different activation functions (tanh, relu, softmax), different optimizers and see the performance
    # Maybe hill climbing algorithm?
    def sigmoid(self, X):
        return 1.0 / (1.0 + np.exp(-X))

    def sigmoid_prime(self, X):
        return self.sigmoid(X) * (1 - self.sigmoid(X))


class NeuralNetwork:
    def __init__(self, input_shape, num_epochs=10, batch_size=32, lr=0.01):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.layers_list = []
        self.input_shape = input_shape
        self.X = None
        self.Y = None
        self.error_history = []

    def compile_network(self):
        self.layers_list[-1].is_output_layer = True
        # Print the total number of parameters and layer wise input output shapes
        print('=' * 50)
        num_parameters = 0
        i = 1
        for layer in self.layers_list:
            layer.init_weights()
            num_parameters += layer.num_neurons * (layer.prev_layer_size + 1)
            print('Layer {}: Input shape is {}, Output shape is {}'
                  ' and weights shape is {}'.format(i, (layer.prev_layer_size, 1), (layer.num_neurons, 1),
                                                    layer.A.shape))
            i += 1

        print('\n')
        print('Total number of parameters are :- ', num_parameters)

        print('=' * 50)
        print('\n')

    def add_layer(self, layer):
        if len(self.layers_list) == 0:
            layer.prev_layer_size = self.input_shape[1]
        else:
            layer.prev_layer_size = self.layers_list[-1].num_neurons

        self.layers_list.append(layer)

    def predict(self, Xp):
        y_hat_all = []
        for x_i in Xp:
            y_hat = x_i.T
            for layer in self.layers_list:
                y_hat = layer.feed_forward(y_hat)
            y_hat_all.append(y_hat[0])
        return self.convert_output(np.array(y_hat_all))

    def predict_raw(self, Xp):
        y_hat = Xp
        for layer in self.layers_list:
            y_hat = layer.feed_forward(y_hat)
        return y_hat

    def evaluate(self, Xe, Ye):
        y_pred = self.predict(Xe)
        assert y_pred.shape == Ye.shape
        return (y_pred == Ye).sum() / len(Ye)

    def update_weights(self, mini_x, mini_y):
        num_samples = len(mini_x)
        loss = 0
        for i in range(num_samples):
            y_hat = self.predict_raw(mini_x[i])
            loss += self.mse_loss(y_hat, mini_y[i])
            for j in range(len(self.layers_list) - 1, -1, -1):
                layer = self.layers_list[j]
                derivative = self.sigmoid_prime(layer.z).reshape(-1, 1)
                # Todo check what's going on after removing the last element from the weight matrix of the next layer
                delta = 2 * (y_hat - mini_y[i]) if layer.is_output_layer \
                    else derivative * (self.layers_list[j + 1].A.T[:-1] @ delta)
                delta = delta.reshape(-1, 1)
                layer.back_prop(delta)

        self.error_history.append(loss / num_samples)

        for layer in self.layers_list[::-1]:
            layer.update_A(self.lr, num_samples)

    def train(self, X, Y):
        self.X = X
        self.Y = Y
        self.compile_network()
        n = len(self.Y)
        for e in range(self.num_epochs):
            p = np.random.permutation(n)
            self.X = self.X[p]
            self.Y = self.Y[p]

            mini_batch_x = []
            mini_batch_y = []
            for k in range(0, n, self.batch_size):
                mini_batch_x.append(self.X[k:k + self.batch_size])
                mini_batch_y.append(self.Y[k:k + self.batch_size])

            num_batches = len(mini_batch_x)
            for i in range(num_batches):
                self.update_weights(mini_batch_x[i], mini_batch_y[i])

            print('Accuracy after epoch {} is {}'.format(e + 1, self.evaluate(self.X, self.Y)))

    def sigmoid(self, X):
        return 1.0 / (1.0 + np.exp(-X))

    def sigmoid_prime(self, X):
        return self.sigmoid(X) * (1 - self.sigmoid(X))

    def convert_output(self, Y):
        return np.where(Y >= 0.5, 1, 0)

    def mse_loss(self, P, Y):
        return ((P - Y) ** 2).sum()


# def fit_logistic(X, Y, learning_rate=0.01, max_iterations=3000, loss_min=0.0001):
#     W = np.full((X.shape[1], Y.shape[1]), 0)
#     b = np.full(Y.shape, 0)
#     loss_array = []
#
#     for i in range(max_iterations):
#         h = sigmoid(np.matmul(X, W) + b)
#         loss = log_loss_2(Y, h)
#         loss_array.append(loss)
#         if loss <= loss_min:
#             break
#         GW = gradient_weights(X, Y, h)
#         GB = gradient_bias(Y, h)
#         W = update_weights(W, learning_rate, GW)
#         b = update_biases(b, learning_rate, GB)
#
#         if i % (max_iterations / 10) == 0:
#             print('Training over iteration... ' + str(i))
#             print(loss)
#     return W, b, loss_array


data = load_breast_cancer()

x = data['data']
x = StandardScaler().fit_transform(x)
y = data['target']
y = y.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(x, y)

neural_network = NeuralNetwork(input_shape=X_train.shape, num_epochs=10)
neural_network.add_layer(HiddenLayer(num_neurons=10))
neural_network.add_layer(HiddenLayer(num_neurons=5))
neural_network.add_layer(HiddenLayer(num_neurons=1))

neural_network.train(X_train, Y_train)

print('\nTest Accuracy is :-', neural_network.evaluate(X_test, Y_test))

# plt.plot(neural_network.error_history)
# plt.show()
