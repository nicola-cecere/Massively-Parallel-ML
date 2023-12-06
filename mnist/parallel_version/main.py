import os
import sys

import matplotlib.pyplot as plt
from kmeans import parallelKMeans
from pyspark import SparkContext
from read import parallelReadFile


def plot_centroids(centroids, image_size=(28, 28)):
    """
    Plot each centroid as an image.

    :param centroids: numpy array, centroids with each row representing a centroid.
    :param image_size: tuple, the (height, width) of the images represented by the centroids.
    """
    for i, centroid in enumerate(centroids):
        # Reshape the centroid to the original image dimensions
        image = centroid.reshape(image_size)

        # Plot the image
        plt.figure()
        plt.imshow(image, cmap="gray")
        plt.title(f"Centroid {i+1}")
        plt.show()


if __name__ == "__main__":
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    sc = SparkContext.getOrCreate()
    current_directory = os.getcwd()
    sc.addPyFile(current_directory + "/MNIST/parallel_version/" + "read.py")
    sc.addPyFile(current_directory + "/MNIST/parallel_version/" + "kmeans.py")
    # read data
    data = parallelReadFile("MNIST/data/tot_mnist_shuf_debug.csv")
    data_cache = data.cache()
    centroids = parallelKMeans(data_cache, 10, 10)
    plot_centroids(centroids)
