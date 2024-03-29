import numpy as np
from pyspark import SparkContext


def train(RDD_Xy, iterations, learning_rate, lambda_reg):
    sc = SparkContext.getOrCreate()

    # Number of features (assuming all records have the same number of features)
    k = len(RDD_Xy.first()[0])
    m = RDD_Xy.count()  # Total number of examples

    np.random.seed(0)  # For reproducibility
    w = np.random.rand(k)  # Weight vector
    b = np.random.rand()  # Bias term

    for i in range(iterations):
        # Broadcast weights and bias
        broadcast_w = sc.broadcast(w)
        broadcast_b = sc.broadcast(b)

        # Compute gradients
        gradients = RDD_Xy.map(
            lambda x: compute_gradients(x, broadcast_w.value, broadcast_b.value, k)
        ).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))

        # Update weights and bias
        w -= learning_rate * ((1 / m) * gradients[0] + (lambda_reg / k) * w)
        b -= learning_rate * (
            (1 / m) * gradients[1] + ((lambda_reg / (2 * k)) * np.sum(w**2))
        )

        # Optional: Print cost for monitoring (not recommended for large datasets)
        cost = RDD_Xy.map(
            lambda x: compute_cost(x, broadcast_w.value, broadcast_b.value, k)
        ).reduce(lambda x, y: x + y)
        cost = (-1 / m) * cost
        cost += (lambda_reg / (2 * k)) * np.sum(w**2)
        print(f"Iteration {i}, Cost: {cost}")

    return w, b


def compute_gradients(record, w, b, k):
    X, y = record
    z = 0
    for i in range(k):
        z += X[i] * w[i]
    z += b
    y_hat = 1 / (1 + np.exp(-z))
    dw = (y_hat - y) * X
    db = y_hat - y
    return dw, db


def compute_cost(record, w, b, k):
    X, y = record
    z = 0
    for i in range(k):
        z += X[i] * w[i]
    z += b
    y_hat = 1 / (1 + np.exp(-z))
    cost = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
    return cost
