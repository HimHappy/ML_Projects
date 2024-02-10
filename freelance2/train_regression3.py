# regularized
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

# Model training script for regression model 1 (Regularized)
model1_regularized = LinearRegression(learning_rate=0.01, batch_size=32, regularization=0.1, max_epochs=100, patience=3)
model1_regularized.fit(X_train[:, [0]], X_train[:, [1]])

# Save model parameters
model1_regularized.save("model3_params.npz")
