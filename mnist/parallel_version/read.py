from pyspark import SparkContext


def is_header(line):
    # Check if the line is a header by a known header feature (e.g., 'label')
    if line.startswith("label"):
        return []
    else:
        return [line]


def convert_to_float(line):
    # Convert all elements except the first one (label) to float and return as a list
    return [float(x) for x in line.split(",")[1:]]


def parallelReadFile(filename):
    """
    Reads a CSV file into an RDD, dropping the first row (header) and the first column (label).

    Parameters:
    filename : str
        The path to the CSV file.

    Returns:
    RDD
        An RDD of data points, where each data point is represented as a list of floats, excluding the label.
    """

    # Get an existing SparkContext or create a new one
    sc = SparkContext.getOrCreate()

    # Read the file into an RDD
    data = sc.textFile(filename)

    # Filter out the header and convert the rest of the lines to float arrays
    processed_data = data.flatMap(is_header).map(convert_to_float)

    return processed_data
