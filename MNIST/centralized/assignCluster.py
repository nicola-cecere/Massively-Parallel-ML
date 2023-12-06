import numpy as np

def serialAssign2cluster(x, centroids):
    distances = np.sqrt(np.sum((centroids - x)**2, axis=1))

    closest_centroid_index = np.argmin(distances)

    return closest_centroid_index
