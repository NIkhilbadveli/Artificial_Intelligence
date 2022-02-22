from math import sqrt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def calc_distance(x1, x2):
    return sqrt(((x1 - x2) ** 2).sum())


def get_k_neighbors_indices(k, train_x, test_point):
    distances = []
    for i in range(len(train_x)):
        distances.append((i, calc_distance(train_x[i], test_point)))

    distances.sort(key=lambda tup: tup[1])  # Using the distance to sort
    distances = distances[1:k + 1]  # Ignoring the first one since it's 0
    return [j for j, d in distances]


def predict_class(k, train_x, train_y, test_point):
    knn_indices = get_k_neighbors_indices(k, train_x, test_point)
    output_values = [train_y[j] for j in knn_indices]
    return max(set(output_values), key=output_values.count)


cancer = datasets.load_breast_cancer()
x_data = cancer.data
y_data = cancer.target

X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data)
K = 1

y_pred = []
for x in X_test:
    y_pred.append(predict_class(K, X_train, Y_train, x))

y_pred = np.array(y_pred)

print('Accuracy with own implementation is :- ', 100 * accuracy_score(Y_test, y_pred))

# Trying with sklearn implementation
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train, Y_train)

y_pred1 = knn.predict(X_test)
print("Accuracy with sklearn's implementation is :- ", 100 * accuracy_score(Y_test, y_pred1))
