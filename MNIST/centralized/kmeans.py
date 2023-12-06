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
    # Initialize centroids
    centroids = initialize_centroids(X, K)
    for _ in range(n_iter):
        # Initialize cluster assignment list
        clusters = [[] for _ in range(K)]
        print("Iteration: ", _)
        for index, sample in X.iterrows():
            closest_centroid_index = serialAssign2cluster(sample, centroids)
            clusters[closest_centroid_index].append(sample)
        # Update centroids
        for i, cluster in enumerate(clusters):
            if len(cluster) != 0:
                centroids[i] = np.mean(cluster, axis=0)
            else:
                centroids[i] = np.random.randn(X.shape[1])
    return centroids
