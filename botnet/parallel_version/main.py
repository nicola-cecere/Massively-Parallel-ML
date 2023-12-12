import os
import sys

from accuracy import accuracy
from normalize import normalize
from preprocess import readFile
from pyspark import SparkContext
from train import train

if __name__ == "__main__":
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    sc = SparkContext.getOrCreate()
    current_directory = os.getcwd()
    sc.addPyFile(current_directory + "/botnet/parallel_version/" + "predict.py")
    sc.addPyFile(current_directory + "/botnet/parallel_version/" + "train.py")

    # read data
    data = readFile("botnet/data/botnet_tot_syn_l.csv")
    # standardize
    data = normalize(data)
    # optimize performance
    data_cached = data.cache()
    # train
    weights, bias = train(data_cached, 10, 1.5, 0.05)
    # accuracy
    accuracy = accuracy(weights, bias, data_cached)
    print("accuracy:", accuracy)
