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
    for x in range(1,1000):
        sample = X.iloc[x]
        closest_centroid_index = serialAssign2cluster(sample, centroids)
        print(f"Data point {x} is closest to centroid {closest_centroid_index}")
    return
