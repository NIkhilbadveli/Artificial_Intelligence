import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd


class GMM:
    def __init__(self, k, max_iter=5):
        self.k = k
        self.max_iter = int(max_iter)

    def initialize(self, X):
        self.shape = X.shape
        self.n, self.m = self.shape

        self.phi = np.full(shape=self.k, fill_value=1 / self.k)
        self.weights = np.full(shape=self.shape, fill_value=1 / self.k)

        random_row = np.random.randint(low=0, high=self.n, size=self.k)
        self.mu = [X[row_index, :] for row_index in random_row]
        self.sigma = [np.cov(X.T) for _ in range(self.k)]

    def e_step(self, X):
        # E-Step: update weights and phi holding mu and sigma constant
        self.weights = self.predict_proba(X)
        self.phi = self.weights.mean(axis=0)

    def m_step(self, X):
        # M-Step: update mu and sigma holding phi and weights constant
        for i in range(self.k):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            self.mu[i] = (X * weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(X.T,
                                   aweights=(weight / total_weight).flatten(),
                                   bias=True)

    def fit(self, X):
        self.initialize(X)

        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)

    def predict_proba(self, X):
        likelihood = np.zeros((self.n, self.k))
        for i in range(self.k):
            distribution = multivariate_normal(
                mean=self.mu[i],
                cov=self.sigma[i])
            likelihood[:, i] = distribution.pdf(X)

        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights

    def predict(self, X):
        weights = self.predict_proba(X)
        return np.argmax(weights, axis=1)


centers = [(0, 4), (5, 5), (8, 2), (20, 10)]
cluster_std = [1.2, 1, 1.1, 2.3]

X, y = make_blobs(n_samples=200, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)

np.random.seed(42)
gmm = GMM(k=4, max_iter=10)
gmm.fit(X)

predicted_clusters = gmm.predict(X)

idx, cluster = [], []
for i, c in enumerate(predicted_clusters):
    idx.append(i)
    cluster.append(c)

plt.figure(figsize=(10, 7))
for clust in np.unique(cluster):
    plt.scatter(X[np.array(cluster) == clust, 0],
                X[np.array(cluster) == clust, 1], s=10, label=f"Cluster{clust}")

plt.legend([f"Cluster {clust}" for clust in np.unique(cluster)], loc="lower right")
plt.title('Clustered Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
