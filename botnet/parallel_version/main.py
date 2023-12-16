import os
import sys

from accuracy import accuracy
from normalize import normalize
from preprocess import readFile
import pyspark
from pyspark import SparkContext
from train import train
import time
import matplotlib.pyplot as plt


if __name__ == "__main__":
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    execution_times = []
    speed_up = []
    for i in range (1,10):
        start_time = time.time()
        number_cores = i
        conf = (
            pyspark.SparkConf()
            .setMaster('local[{}]'.format(number_cores))
        )

        sc = pyspark.SparkContext(conf=conf)
        current_directory = os.getcwd()
        sc.addPyFile(current_directory + "/" + "predict.py")
        sc.addPyFile(current_directory + "/" + "train.py")

        # read data
        data = readFile("../data/botnet_tot_syn_l.csv")
        # standardize
        data = normalize(data)
        # optimize performance
        data_cached = data.cache()
        # train
        weights, bias = train(data_cached, 10, 1.5, 0.05)
        # accuracy
        accuracy_result = accuracy(weights, bias, data_cached)
        print("accuracy:", accuracy_result)
        execution_times.append(time.time() - start_time)
        sc.stop()
    num_workers = list(range(1, len(execution_times)+1))

    # Plotting the graph of execution time
    plt.plot(num_workers, execution_times, marker='o', linestyle='-')
    plt.xlabel('Number of Workers')  # Label for x-axis
    plt.ylabel('Execution Time (s)')  # Label for y-axis
    plt.title('Execution Time vs Number of Workers')  # Title for the plot
    plt.grid(True)  # Show grid lines
    plt.show()

    #Plotting the graph of execution time
    for i in range(0,len(execution_times)):
        speed_up.append(execution_times[0]/execution_times[i])
    plt.plot(num_workers, speed_up, marker='o', linestyle='-')
    plt.xlabel('Number of Workers')  # Label for x-axis
    plt.ylabel('Speed Up')  # Label for y-axis
    plt.title('Speed Up vs Number of Workers')  # Title for the plot
    plt.grid(True)  # Show grid lines
    plt.show()

