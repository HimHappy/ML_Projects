from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from LogisticRegression import LogisticRegression

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
logistic_model1 = LogisticRegression(learning_rate=0.01, batch_size=1, max_epochs=100, patience=3)
logistic_model1.fit(X_train, y_train)
logistic_model1.save("logistic_model1_params.npz")