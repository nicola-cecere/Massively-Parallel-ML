import numpy as np

def train(np_Xy, iterations, learning_rate, lambda_reg):

    # Extracting features (X) and labels (y) from the input array
    X = np.array([example[:-1] for example in np_Xy])  # Extract features
    y = np.array([example[-1] for example in np_Xy])   # Extract labels

    # Number of examples and features
    m, k = X.shape

    np.random.seed(0) # Added for reproducibility
    # Initializing weights and bias with random values
    w = np.random.rand(k)
    b = np.random.rand()

    for i in range(iterations):
        # Compute predictions
        z = np.dot(X, w) + b
        predictions = 1 / (1 + np.exp(-z))  # Applying the sigmoid function

        # Cost function with L2 regularization
        cost = (-1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        cost += (lambda_reg / (2*k)) * np.sum(w**2)

        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {cost}")

        # Gradient calculation
        # dw = (1/m) * np.dot(X.T, (predictions - y)) + (lambda_reg/k) * w
        # db = (1/m) * np.sum(predictions - y)

        # Gradient calculation with np.sum and for loop
        dw = np.zeros(k)
        for j in range(k):
            dw[j] = (1/m) * np.sum((predictions - y) * X[:, j]) + (lambda_reg/k) * w[j]
        db = (1/m) * np.sum(predictions - y)

        # Update weights and bias
        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b


