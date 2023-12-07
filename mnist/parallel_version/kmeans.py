import os

import numpy as np
from assignCluster import parallelAssign2cluster
from pyspark import SparkContext


def initialize_centroids(data, K):
    """
    Initialize K centroids from a standard normal distribution.

    :param data: Pandas DataFrame, the dataset from which feature dimensions are determined.
    :param K: int, the number of centroids to initialize.
    :return: numpy array, initialized centroids.
    """
    num_features = len(data.first())

    # Initialize centroids
    centroids = np.random.randn(K, num_features)

    return centroids


def averageCentroid(x):
    return x[1][0] / x[1][1]


def parallelKMeans(X, K, n_iter):
    sc = SparkContext.getOrCreate()
    current_directory = os.getcwd()
    sc.addPyFile(current_directory + "/MNIST/parallel_version/assignCluster.py")

    # Initialize centroids
    centroids = initialize_centroids(X, K)
    for _ in range(n_iter):
        print("Iteration: ", _)
        # Assign clusters
        broadcast_centroids = sc.broadcast(centroids)
        centroids_assigned = X.map(
            lambda x: parallelAssign2cluster(x, broadcast_centroids.value)
        )
        # Update centroids
        centroids = (
            centroids_assigned.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
            .map(averageCentroid)
            .collect()
        )

    return centroids
