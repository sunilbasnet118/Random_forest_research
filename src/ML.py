import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate Sample Data
# Sample Data for Employees
data = np.array([
    [1, 3, 30, 7],
    [2, 5, 50, 9],
    [1, 3, 20, 6]
])
# Separate features (X) and performance scores (Y)
X = data[:, :-1]
Y = data[:, -1:]


# Normalize the data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y, test_size=0.2, random_state=42)

# Neural Network Parameters
input_size = X.shape[1]
hidden_size = 10
output_size = 1
learning_rate = 0.01
epochs = 1000

# Initialize Weights and Biases
weights_input_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_hidden = np.zeros((1, hidden_size))
bias_output = np.zeros((1, output_size))

# Activation Function (Hyperbolic Tangent)
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

# Training the Neural Network
for epoch in range(epochs):
    # Forward Propagation
    hidden_layer_input = np.dot(X_train, weights_input_hidden) + bias_hidden
    hidden_layer_output = tanh(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = tanh(output_layer_input)

    # Calculate Error
    error = Y_train - predicted_output

    # Backward Propagation
    output_error = error * tanh_derivative(output_layer_input)
    hidden_layer_error = output_error.dot(weights_hidden_output.T) * tanh_derivative(hidden_layer_input)

    # Update Weights and Biases
    weights_hidden_output += hidden_layer_output.T.dot(output_error) * learning_rate
    bias_output += np.sum(output_error, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X_train.T.dot(hidden_layer_error) * learning_rate
    bias_hidden += np.sum(hidden_layer_error, axis=0, keepdims=True) * learning_rate

# Testing the Neural Network
hidden_layer_test = tanh(np.dot(X_test, weights_input_hidden) + bias_hidden)
predicted_scores = tanh(np.dot(hidden_layer_test, weights_hidden_output) + bias_output)

# Evaluate Performance
mse = mean_squared_error(Y_test, predicted_scores)
print(f"Mean Squared Error: {mse}")

# Visualize Results
plt.scatter(Y_test, predicted_scores)
plt.xlabel('Actual Performance Scores')
plt.ylabel('Predicted Performance Scores')
plt.title('Actual vs. Predicted Performance Scores')
plt.show()