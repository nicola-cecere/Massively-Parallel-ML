from kmeans import serialKMeans
from read import serialReadFile
from assignCluster import serialAssign2cluster

if __name__ == "__main__":
    # read data
    data = serialReadFile("/Users/lucapetracca/Documents/GitHub/Massively-Parallel-ML/mnist/data/tot_mnist_shuf.csv")
    centroids = serialKMeans(data, 10, 10)