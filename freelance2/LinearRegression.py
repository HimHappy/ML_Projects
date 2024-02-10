import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self,  learning_rate=0.01, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        learning_rate: float
            The learning rate for gradient descent.
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait for the validation set to decrease.
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def fit(self, X, y, batch_size=None, regularization=None, max_epochs=None, patience=None):
        """Fit a linear model.

        Parameters:
        -----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target values.
        batch_size: int, optional
            The number of samples per batch.
        regularization: float, optional
            The regularization parameter.
        max_epochs: int, optional
            The maximum number of epochs.
        patience: int, optional
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        if batch_size is not None:
            self.batch_size = batch_size
        if regularization is not None:
            self.regularization = regularization
        if max_epochs is not None:
            self.max_epochs = max_epochs
        if patience is not None:
            self.patience = patience

        # Split data into training and validation sets
        num_samples = len(X)
        num_validation = int(0.1 * num_samples)
        X_train, y_train = X[:-num_validation], y[:-num_validation]
        X_val, y_val = X[-num_validation:], y[-num_validation:]

        # Initialize loss_history
        loss_history = []

        # Initialize weights and bias
        self.weights = np.zeros((X_train.shape[1], 1))
        self.bias = 0

        best_weights = self.weights
        best_bias = self.bias
        best_val_loss = float('inf')
        consecutive_increases = 0

        for epoch in range(self.max_epochs):
            for i in range(0, len(X_train), self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]

                # Compute gradients
                gradients, bias_gradient = self.compute_gradients(X_batch, y_batch, self.regularization)

                # Update weights and bias
                self.weights -= self.learning_rate * gradients#.T
                self.bias -= self.learning_rate * bias_gradient

            # Evaluate on validation set
            val_loss = self.mean_squared_error(X_val, y_val)
            loss_history.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.weights
                best_bias = self.bias
                consecutive_increases = 0
            else:
                consecutive_increases += 1
                if consecutive_increases == self.patience:
                    break

        # Set the model parameters to the best ones
        self.weights = best_weights
        self.bias = best_bias

        # Save model parameters
        self.save("model_parameters.npz")

        # Plot loss against step number
        self.loss_history = loss_history
        self.plot_loss(loss_history)

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been trained. Please call fit() first.")

        # Run a forward pass to get predicted values
        predictions = np.dot(X, self.weights) + self.bias
        return predictions

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been trained. Please call fit() first.")

        # Make predictions
        predictions = self.predict(X)

        # Compute mean squared error
        mse = np.mean((y - predictions) ** 2)
        return mse

    def save(self, file_path):
        """Save model parameters to a file.

        Parameters
        ----------
        file_path: str
            The file path to save the parameters.
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been trained. Nothing to save.")

        # Save weights and bias using NumPy's save function
        np.savez(file_path, weights=self.weights, bias=self.bias)

    def load(self, file_path):
        """Load model parameters from a file.

        Parameters
        ----------
        file_path: str
            The file path to load the parameters from.
        """
        # Load weights and bias using NumPy's load function
        data = np.load(file_path)
        self.weights = data['weights']
        self.bias = data['bias']

    def plot_loss(self,loss_history):
        """Plot loss against step number.

        Parameters:
        ----------
        loss_history: list
            A list containing the loss values for each step.
        """
        steps = range(1, len(loss_history) + 1)
        plt.plot(steps, loss_history, marker='o', linestyle='-')
        plt.title('Loss vs Step Number')
        plt.xlabel('Step Number')
        plt.ylabel('Loss')
        plt.show()

    def compute_gradients(self, X, y, regularization):
        """Compute gradients for linear regression with L2 regularization.

        Parameters:
        -----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target values.
        regularization: float
            The regularization parameter.

        Returns:
        --------
        gradients: numpy.ndarray
            Gradients for weights.
        bias_gradient: float
            Gradient for bias.
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been trained. Please call fit() first.")

        # Number of samples
        m = len(X)

        # Predictions
        predictions = np.dot(X, self.weights) + self.bias

        # Compute gradients
        error = predictions - y
        gradients = np.dot(X.T, error) / m
        bias_gradient = np.sum(error) / m

        # Reshape gradients to match the shape of self.weights
        # gradients = gradients.reshape(-1, 1)

        # Add L2 regularization term
        gradients += (regularization / m) * self.weights#.T.squeeze()
        bias_gradient += (regularization / m) * self.bias
        # gradients = gradients.reshape(self.weights.shape) + (regularization / m) * self.weights
        # bias_gradient += (regularization / m) * self.bias


        return gradients, bias_gradient

    def mean_squared_error(self, X, y):
        """Compute mean squared error.

        Parameters:
        -----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.

        Returns:
        --------
        float
            Mean squared error.
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been trained. Please call fit() first.")

        # Make predictions
        predictions = self.predict(X)

        # Compute mean squared error
        mse = np.mean((y - predictions) ** 2)
        return mse
