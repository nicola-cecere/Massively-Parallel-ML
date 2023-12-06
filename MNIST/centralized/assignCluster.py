import numpy as np

def serialAssign2cluster(x, centroids):

    min_distance = np.inf  # Initialize the minimum distance to infinity
    closest_centroid_index = -1  # Initialize the index of the closest centroid

    # Iterate over each centroid to compute the Euclidean distance to the data point x
    for j, centroid in enumerate(centroids):
        centroid = np.array(centroid)  # Ensure centroid is a NumPy array
        distance = np.sqrt(np.sum((x - centroid) ** 2))

        # If the computed distance is less than the current minimum, update the minimum and index
        if distance < min_distance:
            min_distance = distance
            closest_centroid_index = j

    return closest_centroid_index