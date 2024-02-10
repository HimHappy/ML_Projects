# train_regression1.py
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegression

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, stratify=y, random_state=42)

# Model training script for regression model 1 (Non-regularized)
model1 = LinearRegression(learning_rate=0.01, batch_size=32, regularization=0, max_epochs=100, patience=3)
model1.fit(X_train[:, [0]], X_train[:, [1]])

# Save model parameters
model1.save("model1_params.npz")

# # Plot loss
# model1.plot_loss(model1.loss_history)
