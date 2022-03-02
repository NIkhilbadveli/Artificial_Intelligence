from math import log2
import numpy as np


def entropy(values):
    total = len(values)
    unique, counts = np.unique(values, return_counts=True)
    info = 0
    for c in counts:
        p = c / total
        info += -p * log2(p)
    return info


def information_gain(x_data, y_data):
    # Assumes that x_data is feature data for a single column
    N = len(y_data)
    I = entropy(y_data)
    I_avg = 0
    for i in np.unique(y_data):
        child = x_data[y_data == i]
        I_avg += len(child) / N * entropy(child)
    return I - I_avg


def best_split_column(x_data, y_data):
    info_gains = []
    n_rows, n_cols = x_data.shape
    for i in range(n_cols):
        info_gains.append(information_gain(x_data[i], y_data))
    return np.argmax(info_gains)


x = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

# Todo There seems to be some other approach I'm not aware of. What the hell are they doing with left and right nodes.