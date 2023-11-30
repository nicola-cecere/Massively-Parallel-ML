import numpy as np

def predict(w, b, X):

    # Initialize z to 0
    z = 0
    # Iterate over each feature and its corresponding weight
    for i in range(len(w)):
        z += w[i] * X[i]  # Accumulate the product of each feature and its weight
    # Add the bias term to z
    z += b
    # Apply the sigmoid function to get the probability
    p = 1 / (1 + np.exp(-z))
    # Predict the class label (0 or 1) based on the probability
    if p >= 0.5:
        return 1
    else:
        return 0
