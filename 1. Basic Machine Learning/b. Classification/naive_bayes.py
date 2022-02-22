from math import sqrt, pi
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def gaussian_pdf(z, mu, sigma):
    return 1 / (sqrt(2 * pi) * sigma) * np.exp(-(z - mu) ** 2 / (2 * sigma ** 2))


def get_class_pdfs(x_in, y_unique):
    class_pdfs = {}
    for y in y_unique:
        class_pdfs[y] = [(np.mean(x_in[:, i]), np.std(x_in[:, i])) for i in range(x_in.shape[1])]
    return class_pdfs


def calc_probability(z, class_pdfs, class_rows, total_rows):
    # Using log likelihood to avoid underflow
    final_probability = 0
    prior = np.log(class_rows / total_rows)
    final_probability += prior

    for i in range(len(z)):
        final_probability += np.log(gaussian_pdf(z[i], class_pdfs[i][0], class_pdfs[i][1]))

    return final_probability


def make_prediction(x):
    # Using X_train and Y_train while computing
    y_unique = np.unique(Y_train)
    class_pdfs_dict = get_class_pdfs(X_train, y_unique)

    y_pred = []
    for row in x:
        probs = []
        for y in y_unique:
            probs.append((y, calc_probability(row, class_pdfs_dict[y], (Y_train == y).sum(), len(Y_train))))
        y_pred.append(max(probs, key=lambda tup: tup[1])[0])
    return np.array(y_pred)


cancer = datasets.load_breast_cancer()
x_data = cancer.data
y_data = cancer.target

X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data)

# Todo investigate why my accuracy is so low?
print('Accuracy with own implementation is :- ', 100 * accuracy_score(Y_test, make_prediction(X_test)))

# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(X_train, Y_train)

y_pred1 = model.predict(X_test)
print("Accuracy with sklearn's implementation is :- ", 100 * accuracy_score(Y_test, y_pred1))
