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
        # z = np.dot(X, w) + b
        z = np.zeros(m)  # Initialize z as a zero array of length m
        for j in range(k):
            z += w[j] * X[:, j]  # Correctly accumulate the sum of products
        z += b  # Add the bias term to each element of z
        y_hat = 1 / (1 + np.exp(-z))  # Applying the sigmoid function

        # Cost function with L2 regularization
        cost = (-1/m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        cost += (lambda_reg / (2*k)) * np.sum(w**2)

        print(f"Iteration {i}, Cost: {cost}")

        # Gradient calculation
        # dw = (1/m) * np.dot(X.T, (y_hat - y)) + (lambda_reg/k) * w
        # db = (1/m) * np.sum(y_hat - y)

        # Gradient calculation with np.sum and for loop
        dw = np.zeros(k)
        for j in range(k):
            dw[j] = (1/m) * np.sum((y_hat - y) * X[:, j]) + (lambda_reg/k) * w[j]
        db = (1/m) * np.sum(y_hat - y)

        # Update weights and bias
        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b