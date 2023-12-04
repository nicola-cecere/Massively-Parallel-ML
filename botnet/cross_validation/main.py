import os
import sys

from accuracy import accuracy

# from accuracy import accuracy
from get_block_data import get_block_data
from normalize import normalize
from preprocess import readFile
from pyspark import SparkContext
from train import train
from transform import transform

if __name__ == "__main__":
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    sc = SparkContext.getOrCreate()
    current_directory = os.getcwd()
    sc.addPyFile(current_directory + "/botnet/cross_validation/" + "predict.py")
    sc.addPyFile(current_directory + "/botnet/cross_validation/" + "train.py")
    sc.addPyFile(current_directory + "/botnet/cross_validation/" + "transform.py")

    # read data
    data = readFile("botnet/data/botnet_tot_syn_l.csv")
    # standardize
    data = normalize(data)

    num_blocks_cv = 10
    # Shuffle rows and transfrom data
    data_cv = transform(data, num_blocks_cv)
    data_cv_cached = data_cv.cache()

    accuracies = []
    for i in range(num_blocks_cv):
        tr_data, test_data = get_block_data(data_cv_cached, i)
        weights, bias = train(tr_data, 10, 1.5, 0.05)
        acc = accuracy(weights, bias, test_data)
        accuracies.append(acc)
        print("accuracy:", acc)
        print("------------------------------------------------------")
    avg_acc = 0
    for a in accuracies:
        avg_acc += a
    print("average accuracy:", avg_acc / num_blocks_cv)
