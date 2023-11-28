from pyspark import SparkContext
import numpy as np

def normalize(RDD_Xy):
    sc = SparkContext.getOrCreate()

    # Function to compute sum and sum of squares for each feature
    def compute_sum_and_squares(record):
        X, _ = record
        return (np.array(X), np.array(X)**2, 1)

    # Aggregate the sum and sum of squares for each feature, and count the examples
    sum_squares_count = RDD_Xy.map(compute_sum_and_squares).reduce(
        lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]))

    # Calculate the mean and variance for each feature
    mean = sum_squares_count[0] / sum_squares_count[2]
    variance = (sum_squares_count[1] / sum_squares_count[2]) - mean**2
    std_dev = np.sqrt(variance)

    # Replace zeros in standard deviation with ones to avoid division by zero
    std_dev[std_dev == 0] = 1

    # Broadcast the mean and std_dev to all the nodes
    broadcast_mean = sc.broadcast(mean)
    broadcast_std_dev = sc.broadcast(std_dev)

    # Function to normalize features
    def normalize_features(record):
        X, y = record
        X_normalized = (X - broadcast_mean.value) / broadcast_std_dev.value
        return (X_normalized, y)

    # Normalize each feature and return the new RDD
    return RDD_Xy.map(normalize_features)