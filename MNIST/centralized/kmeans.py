import numpy as np
from assignCluster import serialAssign2cluster


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
    X = X.values
    # Initialize centroids
    centroids = initialize_centroids(X, K)
    for _ in range(n_iter):
        # Initialize cluster assignment list
        clusters = {i: [] for i in range(K)}
        print("Iteration: ", _)
        for index, sample in enumerate(X):
            closest_centroid_index = serialAssign2cluster(sample, centroids)
            clusters[closest_centroid_index].append(index)

        # Update centroids
        for i in range(K):
            if clusters[i]:
                assigned_samples = X[clusters[i]]
                centroids[i] = np.mean(assigned_samples, axis=0)
            else:
                centroids[i] = np.random.randn(*centroids[i].shape)
    return centroids
