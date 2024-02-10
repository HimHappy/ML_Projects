from LinearRegression import *

class LogisticRegression:
    def __init__(self, num_features=None, learning_rate=0.01, batch_size=32, max_epochs=100, patience=3):
        """Logistic Regression using Gradient Descent.

        Parameters:
        -----------
        num_features: int, optional
            The number of features in the input data. If not provided, it will be inferred during the first call to fit().
        learning_rate: float
            The learning rate for gradient descent.
        batch_size: int
            The number of samples per batch.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None
        self.num_features = num_features

        if num_features is not None:
            self.weights = np.zeros((num_features, 1))
            self.bias = 0

    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit a logistic regression model.

        Parameters:
        -----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target values (0 or 1).
        X_val: numpy.ndarray, optional
            The validation input data.
        y_val: numpy.ndarray, optional
            The validation target values (0 or 1).
        """
        # Initialize weights and bias
        self.weights = np.zeros((X.shape[1], 1))
        self.bias = 0

        best_weights = self.weights
        best_bias = self.bias
        best_val_loss = float('inf')
        consecutive_increases = 0

        # Initialize loss_history for plotting
        loss_history = []

        for epoch in range(self.max_epochs):
            for i in range(0, len(X), self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                # Compute gradients
                gradients, bias_gradient = self.compute_gradients(X_batch, y_batch)

                # Update weights and bias
                self.weights -= self.learning_rate * gradients
                self.bias -= self.learning_rate * bias_gradient

            # Evaluate on validation set
            val_loss = self.binary_cross_entropy(X_val, y_val) if X_val is not None and y_val is not None else None
            loss_history.append(val_loss)

            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.weights
                best_bias = self.bias
                consecutive_increases = 0
            elif val_loss is not None:
                consecutive_increases += 1
                if consecutive_increases == self.patience:
                    break

        # Set the model parameters to the best ones
        self.weights = best_weights
        self.bias = best_bias

        # Save model parameters
        self.save("model_parameters.npz")

        # Plot loss against step number if loss_history is not empty and does not contain None values
        if loss_history and not all(val is None for val in loss_history):
            self.loss_history = loss_history
            self.plot_loss(loss_history)

    def compute_gradients(self, X, y):
        """Compute gradients for logistic regression."""
        predictions = self.sigmoid(np.dot(X, self.weights) + self.bias)
        error = predictions - y

        # Compute gradients
        gradients = np.dot(X.T, error) / len(X)
        bias_gradient = np.sum(error) / len(X)

        return gradients, bias_gradient

    def binary_cross_entropy(self, X, y):
        """Compute binary cross-entropy loss."""
        predictions = self.sigmoid(np.dot(X, self.weights) + self.bias)
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return loss

    def predict(self, X):
        """Make predictions using the logistic regression model."""
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been trained. Please call fit() first.")

        # Check and handle the case where X has more features than the model was initialized with
        if X.shape[1] > self.weights.shape[0]:
            raise ValueError(f"Input data has {X.shape[1]} features, but the model was initialized with {self.weights.shape[0]} features.")

        # Run a forward pass to get predicted probabilities
        probabilities = self.sigmoid(np.dot(X, self.weights) + self.bias)
        
        # Convert probabilities to binary predictions (0 or 1)
        predictions = (probabilities >= 0.5).astype(int)
        return predictions

    def accuracy(self, X, y):
        """Compute accuracy on the given data.

        Parameters:
        -----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target values (0 or 1).

        Returns:
        --------
        accuracy: float
            The accuracy of the model on the given data.
        """
        predictions = self.predict(X)
        correct_predictions = np.sum(predictions.flatten() == y.flatten())
        total_samples = len(y)
        accuracy = correct_predictions / total_samples
        return accuracy

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

    def plot_loss(self, loss_history):
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
# Example usage:
if __name__ == "__main__":
    # Example data
    X_train_logistic = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train_logistic = np.array([0, 1, 0, 1])

    # Initialize and train the logistic regression model
    logistic_model = LogisticRegression(learning_rate=0.01, batch_size=1, max_epochs=100, patience=3)
    logistic_model.fit(X_train_logistic, y_train_logistic)

    # Make predictions
    X_test_logistic = np.array([[5, 6]])
    predictions_logistic = logistic_model.predict(X_test_logistic)
    print("Logistic Regression Predictions:", predictions_logistic)
