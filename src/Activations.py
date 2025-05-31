import numpy as np

class Activations:
    """Provides various activation functions and their derivatives for a neural network."""

    @staticmethod
    def relu(x):
        """
        ReLU (Rectified Linear Unit) activation function.
        Returns the input itself if it is positive; otherwise, returns 0.
        """
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """
        Derivative of the ReLU function.
        Returns 1 for positive input values, and 0 for negative values.
        """
        return np.where(x > 0, 1, 0)

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function.
        Compresses input values into the range (0, 1).
        Useful in binary classification problems.
        """
        # Clip values to prevent overflow in the exponential calculation
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """
        Derivative of the sigmoid function.
        Important for updating weights during training.
        """
        s = Activations.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def linear(x):
        """
        Linear activation function.
        Returns the input as is. Often used in regression problems.
        """
        return x

    @staticmethod
    def linear_derivative(x):
        """
        Derivative of the linear function.
        Always returns 1, since the slope is constant.
        """
        return np.ones_like(x)
