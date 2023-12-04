# Input RDD_xy
# For loop and shuffle rows
# For each x % num_blocks assign index

from pyspark import SparkContext
import random

def transform(RDD_Xy, num_blocks):
    sc = SparkContext.getOrCreate()

    # Function to assign a random index to each record
    def add_index(record):
        # Randomly select an index between 0 and num_blocks - 1
        index = random.randint(0, num_blocks - 1)
        # Return a new record with the index added
        return (record, index)

    # Map each record in the RDD to include a random fold index
    RDD_tranformed = RDD_Xy.map(add_index)
    return RDD_tranformed




