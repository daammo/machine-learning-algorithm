from linear_regression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate a simple linear dataset
X, y = make_regression(n_samples=1000, n_features=1, noise=20, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the linear regression model
regressor = LinearRegression(learning_rate=0.01, n_iters=1000)

# Train the model on the training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
predictions = regressor.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Plotting the test data points
plt.scatter(X_test, y_test, color='blue', label='Actual')

# Plotting the predicted data points
plt.plot(X_test, predictions, color='red', label='Predicted')

plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()
