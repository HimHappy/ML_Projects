from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Spliting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and load the logistic regression model1
model1 = LogisticRegression(num_features=4)
model1.load("logistic_model1_params.npz")

# Evaluate model1 on the test set
accuracy_model1 = model1.accuracy(X_test, y_test)
print("Model 1 Test Accuracy:", accuracy_model1)
