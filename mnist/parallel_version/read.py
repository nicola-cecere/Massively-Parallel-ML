from pyspark import SparkContext

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
    processed_data = data.map(lambda line: [float(x) for x in line.split(",")[1:]])

    return processed_data