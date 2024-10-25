import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the provided dataset
file_path = "C:/Year 3/Regression/Nairobi Office Price Ex.csv"
df = pd.read_csv(file_path)

# Extracting the relevant data
x = df['SIZE'].values
y = df['PRICE'].values

# Initialize parameters
m_initial = np.random.rand()  # Random initial value for slope (m)
c_initial = np.random.rand()  # Random initial value for y-intercept (c)
learning_rate = 0.0001
epochs = 10

# Function to compute Mean Squared Error (MSE)
def compute_mse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

# Function for Gradient Descent
def gradient_descent(x, y, m, c, learning_rate, epochs):
    n = len(x)
    mse_history = []

    for epoch in range(epochs):
        y_pred = m * x + c
        m_gradient = (-2/n) * np.sum(x * (y - y_pred))
        c_gradient = (-2/n) * np.sum(y - y_pred)

        m -= learning_rate * m_gradient
        c -= learning_rate * c_gradient

        mse = compute_mse(y, y_pred)
        mse_history.append(mse)
        print(f"Epoch {epoch + 1}: MSE = {mse:.4f}")

    return m, c, mse_history

# Train the model
m_final, c_final, mse_history = gradient_descent(x, y, m_initial, c_initial, learning_rate, epochs)

# Plot the result
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, m_final * x + c_final, color='red', label='Line of Best Fit')
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Office Price')
plt.title('Linear Regression - Line of Best Fit')
plt.legend()
plt.show()

# Predict the price for a 100 sq. ft. office
size_100 = 100
predicted_price_100 = m_final * size_100 + c_final
print(f"Predicted price for a 100 sq. ft. office: {predicted_price_100}")
