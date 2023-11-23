"""
Logistic Loss Component:

- The expression y * np.log(predictions) + (1 - y) * np.log(1 - predictions) calculates the logistic loss for each training example.
- For a single training example with label y and predicted probability predictions, the logistic loss is computed as:
  - y * np.log(predictions) if y = 1
  - (1 - y) * np.log(1 - predictions) if y = 0
- This formula reflects the negative log-likelihood of observing the given label (either 0 or 1) for each example, based on the predicted probability.

Summing Over All Examples:

- np.sum(...) sums this loss over all training examples.
- This sum gives the total logistic loss across the entire training dataset.

Average Loss:

- The total loss is then divided by m, the number of training examples ((-1/m) * np.sum(...)), to compute the average logistic loss per example.
- Averaging is important to ensure that the cost doesn't inherently increase with the number of examples.

Negative Sign:

- The negative sign (-1/m) in front of the sum converts the log likelihood into a cost (or loss) that we want to minimize. Higher probabilities assigned to the correct labels will lead to a lower cost.

Cost function with L2 regularization
"""

# cost = (-1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
#cost += (lambda_reg / (2*k)) * np.sum(w**2)

"""
Gradient of Logistic Loss:

- np.dot(X.T, (predictions - y)) performs a matrix multiplication between the transpose of X (features matrix) and the vector (predictions - y).
- (predictions - y) is the difference between the predicted probabilities and the actual labels for all training examples.
- By transposing X and performing this dot product, we effectively sum the contributions of each training example to the gradient of each weight. This is equivalent to applying np.sum over the dataset but in a vectorized and more efficient manner.

Regularization Term:

- (lambda_reg/k) * w is the gradient of the L2 regularization term. It is added to the gradient of the logistic loss.
- k is used in the denominator, in accordance with your PDF, to scale the regularization term based on the number of features.

Importance of np.dot:

- The use of np.dot(X.T, ...) in the computation of dw is a vectorized way to calculate the sum of the gradients for each feature across all training examples. It’s more efficient than looping through each example and feature.
- This matrix operation takes care of the summation implicitly, making the code concise and computationally efficient, especially for large datasets.
"""
# Gradient calculation
# dw = (1/m) * np.dot(X.T, (predictions - y)) + (lambda_reg/k) * w
# db = (1/m) * np.sum(predictions - y)