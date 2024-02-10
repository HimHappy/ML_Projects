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

# Evaluate model on the test set
model2 = LinearRegression()
model2.load("model1_regularized_params.npz")
test_score2 = model2.score(X_test[:, [0]], X_test[:, [1]])
print("Model 2 Test Score:", test_score2)
