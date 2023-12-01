import numpy as np
from pyspark import SparkContext


def train(RDD_Xy, iterations, learning_rate, lambda_reg):
    sc = SparkContext.getOrCreate()

    # Number of features (assuming all records have the same number of features)
    num_features = len(RDD_Xy.first()[0])
    m = RDD_Xy.count()  # Total number of examples

    np.random.seed(0)  # For reproducibility
    w = np.random.rand(num_features)  # Weight vector
    b = np.random.rand()  # Bias term

    for i in range(iterations):
        # Broadcast weights and bias
        broadcast_w = sc.broadcast(w)
        broadcast_b = sc.broadcast(b)

        # Compute gradients
        gradients = RDD_Xy.map(
            lambda x: compute_gradients(
                x, broadcast_w.value, broadcast_b.value, m, lambda_reg
            )
        ).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))

        # Update weights and bias
        w -= learning_rate * gradients[0]
        b -= learning_rate * gradients[1]

        # Optional: Print cost for monitoring (not recommended for large datasets)
        cost = RDD_Xy.map(
            lambda x: compute_cost(
                x, broadcast_w.value, broadcast_b.value, lambda_reg, m
            )
        ).reduce(lambda x, y: x + y)
        print(f"Iteration {i}, Cost: {cost}")

    return w, b


def compute_gradients(point, w, b, m, lambda_reg):
    X, y = point
    z = np.dot(X, w) + b
    y_hat = 1 / (1 + np.exp(-z))
    dw = (1 / m) * np.dot(X, (y_hat - y)) + (lambda_reg / m) * w
    db = (1 / m) * np.sum(y_hat - y)
    return dw, db


def compute_cost(point, w, b, lambda_reg, m):
    X, y = point
    z = np.dot(X, w) + b
    y_hat = 1 / (1 + np.exp(-z))
    cost = (-1 / m) * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    cost += (lambda_reg / (2 * m)) * np.sum(w**2)
    return cost
