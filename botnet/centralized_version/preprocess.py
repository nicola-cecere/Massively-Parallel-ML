import os

import numpy as np


def readFile(filename):
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    data = []
    with open(parent_directory + "/" + filename, "r") as file:
        for line in file:
            # Split the line into columns, convert each to float, and add to data list
            # Note: The last column (label) is also converted to float for consistency in the array
            cols = [float(x) for x in line.strip().split(",")]
            data.append(cols)

    data_array = np.array(data)

    return data_array
