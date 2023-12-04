def get_block_data(data_cv, block_numb):
    """
    Splits the input RDD into two RDDs based on the index value.

    :param data_cv: An RDD where each row is a list of size 11, y, and index.
    :param block_numb: The block number to split the RDD.
    :return: A tuple of two RDDs (tr_data, test_data).
    """

    # Split the data into two RDDs based on the index value

    def filter_block_train(record):
        if record[1] != block_numb:
            return [record[0]]
        else:
            return []

    def filter_block_test(record):
        if record[1] == block_numb:
            return [record[0]]
        else:
            return []

    tr_data = data_cv.flatMap(filter_block_train)
    test_data = data_cv.flatMap(filter_block_test)

    return tr_data, test_data
