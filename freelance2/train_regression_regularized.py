from LinearRegression import *
from train_regression1 import *
model_regularized = LinearRegression(learning_rate=0.01, batch_size=32, regularization=0.1, max_epochs=100, patience=3)
model_regularized.fit(X_train[:, [0]], X_train[:, [1]])

# Save model parameters
model_regularized.save("model_regularized_params.npz")

# Plot loss
model_regularized.plot_loss(model_regularized.loss_history)

# Compare weights between regularized and non-regularized models
print("Non-Regularized Model Weights:", model1.weights)
print("Regularized Model Weights:", model_regularized.weights)
