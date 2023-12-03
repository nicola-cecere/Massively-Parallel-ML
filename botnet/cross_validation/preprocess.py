import os

from pyspark import SparkContext


def readFile(filename):
    """
    Return an RDD containing the data of filename.
    Each example (row) of the file corresponds to one RDD record.
    Each record of the RDD is a tuple (X,y). “X” is an array containing the 11
    features (float number) of an example
    “y” is the 12th column of an example (integer 0/1)
    """

    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    sc = SparkContext.getOrCreate()
    data = sc.textFile(current_directory + "/" + filename)
    processed_data = data.map(lambda line: line.split(",")).map(
        lambda cols: ([float(x) for x in cols[:11]], float(cols[11]))
    )
    return processed_data
