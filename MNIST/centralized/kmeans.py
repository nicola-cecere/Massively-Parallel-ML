import numpy as np


def initialize_centroids(data, K):
    """
    Initialize K centroids from a standard normal distribution.

    :param data: Pandas DataFrame, the dataset from which feature dimensions are determined.
    :param K: int, the number of centroids to initialize.
    :return: numpy array, initialized centroids.
    """
    # Number of features in the data
    num_features = data.shape[1]

    # Initialize centroids
    centroids = np.random.randn(K, num_features)

    return centroids


def serialKMeans(X, K, n_iter):
    # Initialize centroids
    centroids = initialize_centroids(X, K)

    return
