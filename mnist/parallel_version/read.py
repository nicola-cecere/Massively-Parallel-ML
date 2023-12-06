from pyspark import SparkContext


def convert_to_float(line):
    for x in line.split(",")[1:]:
        if x == "1x1":
            return []
        else:
            return [float(x)]


def parallelReadFile(filename):
    """
    Reads a CSV file into an RDD, dropping the first column (label).

    Parameters:
    filename : str
        The path to the CSV file.

    Returns:
    RDD
        An RDD of data points, where each data point is represented as a list of floats.
    """

    # Get an existing SparkContext or create a new one
    sc = SparkContext.getOrCreate()

    # Read the file into an RDD
    data = sc.textFile(filename)

    # Split each line by comma, convert to float, and drop the first element (label)
    processed_data = data.flatMap(convert_to_float)

    return processed_data
