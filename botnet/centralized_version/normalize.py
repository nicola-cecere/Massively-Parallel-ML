import numpy as np
def normalize(np_Xy):

    # Extracting features (X) and labels (y) from the input array
    X = np_Xy[:, :-1]
    y = np_Xy[:, -1].reshape(-1, 1)

    # Computing mean and standard deviation
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)

    # Avoid division by zero
    std_dev[std_dev == 0] = 1

    # Compute normalized features and stack the labels
    X_norm = (X - mean) / std_dev
    X_norm = np.hstack((X_norm, y))

    return X_norm

