import numpy as np

class MultiOutputRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Multi-Output Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait for the validation set to decrease.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Fit a multi-output regression model.

        Parameters:
        -----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        # Split data into training and validation sets
        num_samples = len(X)
        num_validation = int(0.1 * num_samples)
        X_train, y_train = X[:-num_validation], y[:-num_validation]
        X_val, y_val = X[-num_validation:], y[-num_validation:]

        # Initialize weights and bias
        self.weights = np.zeros((X_train.shape[1], y_train.shape[1]))
        self.bias = np.zeros((1, y_train.shape[1]))

        best_weights = self.weights
        best_bias = self.bias
        best_val_loss = float('inf')
        consecutive_increases = 0

        for epoch in range(max_epochs):
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Compute gradients
                gradients, bias_gradient = self.compute_gradients(X_batch, y_batch, regularization)

                # Update weights and bias
                self.weights -= self.learning_rate * gradients
                self.bias -= self.learning_rate * bias_gradient

            # Evaluate on validation set
            val_loss = self.mean_squared_error(X_val, y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.weights
                best_bias = self.bias
                consecutive_increases = 0
            else:
                consecutive_increases += 1
                if consecutive_increases == patience:
                    break

        # Set the model parameters to the best ones
        self.weights = best_weights
        self.bias = best_bias

    def compute_gradients(self, X, y, regularization):
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been trained. Please call fit() first.")

        # Number of samples
        n = len(X)

        # Predictions
        predictions = self.predict(X)

        # Compute gradients
        error = predictions - y
        gradients = np.dot(X.T, error) / n
        bias_gradient = np.sum(error, axis=0) / n

        # regularization term
        gradients += (regularization / n) * self.weights
        bias_gradient += (regularization / n) * self.bias

        return gradients, bias_gradient

    def mean_squared_error(self, X, y):
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been trained. Please call fit() first.")

        # Number of samples and outputs
        n, m = y.shape

        # Predictions
        predictions = self.predict(X)

        # Compute mean squared error
        mse = np.sum((y - predictions) ** 2) / (n * m)
        return mse

    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been trained. Please call fit() first.")

        # Run a forward pass to get predicted values
        predictions = np.dot(X, self.weights) + self.bias
        return predictions

    def score(self, X, y):
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been trained. Please call fit() first.")

        predictions = self.predict(X)

        # mean squared error
        mse = np.mean((y - predictions) ** 2)
        return mse

    def save(self, file_path):
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been trained. Nothing to save.")

        # Save weights and bias 
        np.savez(file_path, weights=self.weights, bias=self.bias)

    def load(self, file_path):
        data = np.load(file_path)
        self.weights = data['weights']
        self.bias = data['bias']