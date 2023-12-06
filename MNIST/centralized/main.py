import matplotlib.pyplot as plt
from assignCluster import serialAssign2cluster
from kmeans import serialKMeans
from read import serialReadFile


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
    # read data
    data = serialReadFile("MNIST/data/tot_mnist_shuf.csv")
    centroids = serialKMeans(data, 10, 10)
    plot_centroids(centroids)
