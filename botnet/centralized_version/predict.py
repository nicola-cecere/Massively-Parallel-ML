import numpy as np

def predict(w, b, X):
<<<<<<< HEAD

    # Initialize z to 0
    z = 0
    # Iterate over each feature and its corresponding weight
    for i in range(len(w)):
        z += w[i] * X[i]  # Accumulate the product of each feature and its weight
    # Add the bias term to z
    z += b
=======
    # Initialize the sum
    z = 0
    # Iterate over the weights and corresponding features
    for i in range(len(w)):
        z += w[i] * X[i]
    # Add the bias term
    z += b

    # compact way to calculate it z = np.dot(w, X) + b

>>>>>>> e7eea3858c131c77c38dd9f78bddc2da2cd0b8c4
    # Apply the sigmoid function to get the probability
    p = 1 / (1 + np.exp(-z))
    # Predict the class label (0 or 1) based on the probability
    if p >= 0.5:
        return 1
    else:
        return 0
